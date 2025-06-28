.. SPDX-License-Identifier: AGPL-3.0-or-later

SearXNG AI Kit
==============

**This is a fork of SearXNG focused on providing an AI-powered search toolkit.**

**AI-powered search for your terminal, code, and AI assistants**

🔍 **CLI tool** - Search from your terminal with AI-enhanced research capabilities

🤖 **AI chat** - Ask questions and get comprehensive research using 180+ search engines

🐍 **Python library** - Programmatic search and AI-powered content retrieval

🔌 **MCP server** - AI assistant integration with advanced research tools

📦 **Standalone binaries** - Pre-built executables for Linux, Windows, and macOS

⚡ **Fast and private** - No tracking, no ads, just intelligent results

**Installation:**

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/nikvdp/searxng-ai-kit
   cd searxng-ai-kit
   
   # Install from source
   pip install .
   
   # Or with uv
   uv pip install .

**Quick Start:**

.. code-block:: bash

   # AI Chat: Research with intelligent analysis
   searxng chat "What are the latest developments in AI?"
   
   # CLI: Search with multiple engines
   searxng search "machine learning" --engines duckduckgo,startpage
   
   # JSON output for scripting
   searxng search "python tutorial" --format json
   
   # Library: Use in Python code
   import searxng
   results = searxng.search("python tutorial")
   
   # MCP Server: For AI assistant integration
   searxng mcp-server

📖 **For complete documentation, see:** `README.cli.md <README.cli.md>`_

----

**Original SearXNG Documentation Below**

.. figure:: https://raw.githubusercontent.com/searxng/searxng/master/client/simple/src/brand/searxng.svg
   :target: https://docs.searxng.org/
   :alt: SearXNG
   :width: 100%
   :align: center

----

Privacy-respecting, hackable `metasearch engine`_

Searx.space_ lists ready-to-use running instances.

A user_, admin_ and developer_ handbook is available on the homepage_.

|SearXNG install|
|SearXNG homepage|
|SearXNG wiki|
|AGPL License|
|Issues|
|commits|
|weblate|
|SearXNG logo|

----

.. _searx.space: https://searx.space
.. _user: https://docs.searxng.org/user
.. _admin: https://docs.searxng.org/admin
.. _developer: https://docs.searxng.org/dev
.. _homepage: https://docs.searxng.org/
.. _metasearch engine: https://en.wikipedia.org/wiki/Metasearch_engine

.. |SearXNG logo| image:: https://raw.githubusercontent.com/searxng/searxng/master/client/simple/src/brand/searxng-wordmark.svg
   :target: https://docs.searxng.org/
   :width: 5%

.. |SearXNG install| image:: https://img.shields.io/badge/-install-blue
   :target: https://docs.searxng.org/admin/installation.html

.. |SearXNG homepage| image:: https://img.shields.io/badge/-homepage-blue
   :target: https://docs.searxng.org/

.. |SearXNG wiki| image:: https://img.shields.io/badge/-wiki-blue
   :target: https://github.com/searxng/searxng/wiki

.. |AGPL License|  image:: https://img.shields.io/badge/license-AGPL-blue.svg
   :target: https://github.com/searxng/searxng/blob/master/LICENSE

.. |Issues| image:: https://img.shields.io/github/issues/searxng/searxng?color=yellow&label=issues
   :target: https://github.com/searxng/searxng/issues

.. |PR| image:: https://img.shields.io/github/issues-pr-raw/searxng/searxng?color=yellow&label=PR
   :target: https://github.com/searxng/searxng/pulls

.. |commits| image:: https://img.shields.io/github/commit-activity/y/searxng/searxng?color=yellow&label=commits
   :target: https://github.com/searxng/searxng/commits/master

.. |weblate| image:: https://translate.codeberg.org/widgets/searxng/-/searxng/svg-badge.svg
   :target: https://translate.codeberg.org/projects/searxng/


Contact
=======

Ask questions or chat with the SearXNG community (this not a chatbot) on

IRC
  `#searxng on libera.chat <https://web.libera.chat/?channel=#searxng>`_
  which is bridged to Matrix.

Matrix
  `#searxng:matrix.org <https://matrix.to/#/#searxng:matrix.org>`_


Setup
=====

- A well maintained `Docker image`_, also built for ARM64 and ARM/v7
  architectures.
- Alternatively there are *up to date* `installation scripts`_.
- For individual setup consult our detailed `Step by step`_ instructions.
- To fine-tune your instance, take a look at the `Administrator documentation`_.

.. _Administrator documentation: https://docs.searxng.org/admin/index.html
.. _Step by step: https://docs.searxng.org/admin/installation-searxng.html
.. _installation scripts: https://docs.searxng.org/admin/installation-scripts.html
.. _Docker image: https://github.com/searxng/searxng-docker

Translations
============

.. _Weblate: https://translate.codeberg.org/projects/searxng/searxng/

Help translate SearXNG at `Weblate`_

.. figure:: https://translate.codeberg.org/widgets/searxng/-/multi-auto.svg
   :target: https://translate.codeberg.org/projects/searxng/


Contributing
============

.. _development quickstart: https://docs.searxng.org/dev/quickstart.html
.. _developer documentation: https://docs.searxng.org/dev/index.html

Are you a developer?  Have a look at our `development quickstart`_ guide, it's
very easy to contribute.  Additionally we have a `developer documentation`_.


Codespaces
==========

You can contribute from your browser using `GitHub Codespaces`_:

- Fork the repository
- Click on the ``<> Code`` green button
- Click on the ``Codespaces`` tab instead of ``Local``
- Click on ``Create codespace on master``
- VSCode is going to start in the browser
- Wait for ``git pull && make install`` to appear and then disappear
- You have `120 hours per month`_ (see also your `list of existing Codespaces`_)
- You can start SearXNG using ``make run`` in the terminal or by pressing ``Ctrl+Shift+B``

.. _GitHub Codespaces: https://docs.github.com/en/codespaces/overview
.. _120 hours per month: https://github.com/settings/billing
.. _list of existing Codespaces: https://github.com/codespaces
