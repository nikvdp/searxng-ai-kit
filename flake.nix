{
  description = "SearXNG AI Kit: CLI, Python library, and MCP server";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
    };
  };

  outputs = { nixpkgs, pyproject-nix, uv2nix, pyproject-build-systems, ... }:
    let
      inherit (nixpkgs) lib;
      forSystems = lib.genAttrs [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
      # Keep setup.py as the authority for the upstream revision.
      upstreamCommit = builtins.head (builtins.match ".*SEARXNG_COMMIT = \"([a-f0-9]+)\".*" (builtins.readFile ./setup.py));
      perSystem = forSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          python = pkgs.python312;
          upstreamSource = pkgs.fetchgit {
            url = "https://github.com/searxng/searxng.git";
            rev = upstreamCommit;
            leaveDotGit = true;
            hash = "sha256-W8NzHuFD/z/Qp/etVS3VEtWF7kzBn1iuj9FWpiFdhWo=";
          };
          upstreamWheel = pkgs.stdenv.mkDerivation {
            pname = "searxng-wheel";
            version = upstreamCommit;
            src = upstreamSource;
            nativeBuildInputs = [
              pkgs.git
              pkgs.uv
              (python.withPackages (ps: [ ps.setuptools ps.wheel ps.pyyaml ps.msgspec ps.typing-extensions ]))
            ];
            UV_PYTHON_DOWNLOADS = "never";
            UV_OFFLINE = "1";
            buildPhase = ''
              runHook preBuild
              export HOME="$TMPDIR"
              export GITHUB_REPOSITORY=searxng/searxng
              export GITHUB_REF_NAME=master
              uv run --no-project --no-sync python -m searx.version freeze
              uv build --wheel --no-build-isolation --out-dir "$out"
              runHook postBuild
            '';
            dontInstall = true;
          };
          pythonSet = (pkgs.callPackage pyproject-nix.build.packages { inherit python; }).overrideScope
            (lib.composeManyExtensions [
              pyproject-build-systems.overlays.wheel
              (workspace.mkPyprojectOverlay { sourcePreference = "wheel"; })
              (final: prev: {
                resolveBuildSystem = spec:
                  prev.resolveBuildSystem (builtins.removeAttrs spec [ "uv" ])
                  ++ lib.optional (spec ? uv) pkgs.uv;
                searxng-ai-kit = prev.searxng-ai-kit.overrideAttrs (old: {
                  src = lib.fileset.toSource {
                    root = ./.;
                    fileset = lib.fileset.unions [
                      ./setup.py ./build_searxng_wheel.py ./pyproject.toml
                      ./pyproject.toml.template ./README.md ./searxng_cli.py ./searxng
                    ];
                  };
                  # Seed the existing wheel cache; setup.py still performs all vendoring.
                  preBuild = (old.preBuild or "") + ''
                    export HOME="$TMPDIR"
                    export UV_PYTHON="${python.interpreter}"
                    mkdir -p wheels
                    cp ${upstreamWheel}/*.whl wheels/
                    uv run --no-project --no-sync --offline python -c 'from pathlib import Path; from build_searxng_wheel import save_build_metadata; save_build_metadata(next(Path("wheels").glob("*.whl")), "${upstreamCommit}", "wheels")'
                  '';
                  UV_PYTHON_DOWNLOADS = "never";
                  UV_OFFLINE = "1";
                });
              })
            ]);
          environment = pythonSet.mkVirtualEnv "searxng-ai-kit-env" workspace.deps.default;
          inherit (pkgs.callPackages pyproject-nix.build.util { }) mkApplication;
        in {
          inherit pkgs python upstreamWheel environment;
          application = mkApplication { venv = environment; package = pythonSet.searxng-ai-kit; };
        });
    in {
      packages = forSystems (system: {
        default = perSystem.${system}.application;
        searxng-ai-kit = perSystem.${system}.application;
        python-env = perSystem.${system}.environment;
        upstream-wheel = perSystem.${system}.upstreamWheel;
      });
      apps = forSystems (system: {
        default = {
          type = "app";
          program = "${perSystem.${system}.application}/bin/searxng";
          meta.description = "SearXNG AI Kit";
        };
      });
      checks = forSystems (system:
        let inherit (perSystem.${system}) pkgs environment application upstreamWheel;
        in {
          installed = pkgs.runCommand "searxng-installed-check" {
            nativeBuildInputs = [ pkgs.uv ];
          } ''
            export HOME="$TMPDIR"
            export UV_PYTHON="${environment}/bin/python"
            export UV_PYTHON_DOWNLOADS=never
            export UV_OFFLINE=1
            cd "$TMPDIR"
            ${application}/bin/searxng --help
            ${application}/bin/searxng engines --common
            uv run --no-project python -I ${./tests/test_http_server.py}
            uv run --no-project python -I -c '
            from pathlib import Path
            import zipfile
            import searx
            import searxng
            root = Path(searx.__file__).parent.parent
            with zipfile.ZipFile(next(Path("${upstreamWheel}").glob("*.whl"))) as wheel:
                for name in wheel.namelist():
                    if name.startswith("searx/") and not name.endswith("/"):
                        assert (root / name).read_bytes() == wheel.read(name), name
            '
            touch "$out"
          '';
        });
      devShells = forSystems (system: {
        default = perSystem.${system}.pkgs.mkShell {
          packages = [ perSystem.${system}.environment perSystem.${system}.pkgs.uv perSystem.${system}.pkgs.git ];
          UV_NO_SYNC = "1";
          UV_PYTHON_DOWNLOADS = "never";
        };
      });
    };
}
