# CMind: An AI Agent for Localizing C Memory Bugs

Proposed by 
- [Chia-Yi Su](https://chiayisu.github.io/)
- [Collin McMillan](https://sdf.org/~cmc/)

## Quick link
- [Introduction](#introduction)
- [To-do list](#to-do-list)
- [How to run it](#how-to-run-it)
- [Citation](#citation)
- [Bug Reports](#bug-reports)
- [Examples](#examples)


## Introduction

This repository contains the code and the insttuctions to build Cmind locally. We also release a [Document](https://apclbuglocalizer.github.io/) with more detailed information on how Cmind works. If you prefer using our website for bug localization, you can also access it via [Link](http://172.237.148.235/).

## To-do list

- Please install [Docker](https://www.docker.com/get-started/) as this requires docker to install some dependencies. You will not need to install any dependency including Python packages. When you run it, it will install all dependencies automatically.

- Set-up the path in ```config.json```. The parameters are as follows:
```
{
    "doxyfile_dir": "Path of Doxyfile for generating callgraph",
    "model_name": "GPT models tag",
    "joern_dir": "Joern directory e.g. '/opt/joern/joern-cli/joern'. Note that it needs to be the directory in the docker container not local directory"
    "_joern_dir": "Joern directory for tesing locally e.g. '/home/chiayi/bin/joern/joern-cli/joern' You will need to change it to joern_dir yourself when run it locally for debug",
    "openai_key": "Path for your OpenAI key",
    "project_dir": "Dirctory of your source code",
    "report_file": "Directory for your bug report e.g. './bug_report.txt'",
    "container_name": "The name of your docker container e.g. debugger_container"
}
```

## How to run it 
``` 
python3 run.py --config-file=config.json
```
```
--config-file: Path of your config.json file
```
Note that before you run the script, please make sure all paths are correct.
## Bug Reports
The following table is the link to each bug report with PR if any. ``o`` means Cmind successfully localizes the bugs. ``x`` means Cmind fails to localize the bugs. The results and source code are in [Link](results/).


| #  | Issue                                                                 | PR / Commit                                                                                             | gpt-o4 | gpt-5 | Results                  |
|----|-----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|:------:|:-----:|--------------------------|
| 1  | [redis#14188](https://github.com/redis/redis/issues/14188)            | [redis#14191](https://github.com/redis/redis/pull/14191)                                                | ✅     | ✅    | [link](results/result1)  |
| 2  | [obs-studio#11888](https://github.com/obsproject/obs-studio/issues/11888) | N/A                                                                                                     | ✅     | ✅    | [link](results/result2)  |
| 3  | [obs-studio#11457](https://github.com/obsproject/obs-studio/issues/11457) | N/A                                                                                                     | ✅     | ✅    | [link](results/result3)  |
| 4  | [obs-studio#10512](https://github.com/obsproject/obs-studio/issues/10512) | N/A                                                                                                     | ✅     | ✅    | [link](results/result4)  |
| 5  | [masscan#730](https://github.com/robertdavidgraham/masscan/issues/730) | [masscan@66c5159](https://github.com/robertdavidgraham/masscan/commit/66c5159)                         | ✅     | ✅    | [link](results/result5)  |
| 6  | [masscan#391](https://github.com/robertdavidgraham/masscan/issues/391) | [masscan#390](https://github.com/robertdavidgraham/masscan/pull/390)                                   | ✅     | ✅    | [link](results/result6)  |
| 7  | [zephyr#92388](https://github.com/zephyrproject-rtos/zephyr/issues/92388) | N/A                                                                                                     | ❌     | ✅    | [link](results/result7)  |
| 8  | [zephyr#92601](https://github.com/zephyrproject-rtos/zephyr/issues/92601) | [zephyr#92996](https://github.com/zephyrproject-rtos/zephyr/pull/92996)                                 | ✅     | ✅    | [link](results/result8)  |
| 9  | [Sandboxie#4658](https://github.com/sandboxie-plus/Sandboxie/issues/4658) | N/A                                                                                                     | ✅     | ✅    | [link](results/result9)  |
| 10 | [Sandboxie#4267](https://github.com/sandboxie-plus/Sandboxie/issues/4267) | [Sandboxie#4268](https://github.com/sandboxie-plus/Sandboxie/pull/4268)                                 | ✅     | ✅    | [link](results/result10) |
| 11 | [libvips#4352](https://github.com/libvips/libvips/issues/4352)        | [libvips@8e7bdde](https://github.com/libvips/libvips/commit/8e7bdde)                                     | ❌     | ❌    | [link](results/result11) |
| 12 | [libvips#4170](https://github.com/libvips/libvips/issues/4170)        | [libvips#4474](https://github.com/libvips/libvips/pull/4474)                                             | ❌     | ❌    | [link](results/result12) |
| 13 | [mgba#3452](https://github.com/mgba-emu/mgba/issues/3452)            | [mgba@86453b8](https://github.com/mgba-emu/mgba/commit/86453b8)                                         | ✅     | ✅    | [link](results/result13) |
| 14 | [libvips#4104](https://github.com/libvips/libvips/issues/4104)        | [libvips#4105](https://github.com/libvips/libvips/pull/4105)                                             | ✅     | ✅    | [link](results/result14) |
| 15 | [libvips#3984](https://github.com/libvips/libvips/issues/3984)        | [libvips#3990](https://github.com/libvips/libvips/pull/3990)                                             | ❌     | ❌    | [link](results/result15) |
| 16 | [htop#1449](https://github.com/htop-dev/htop/issues/1449)            | [htop#1450](https://github.com/htop-dev/htop/pull/1450)                                                  | ✅     | ✅    | [link](results/result16) |
| 17 | [htop#1662](https://github.com/htop-dev/htop/issues/1662)            | N/A                                                                                                     | ✅     | ✅    | [link](results/result17) |
| 18 | [redis#14184](https://github.com/redis/redis/issues/14184)            | [redis#14186](https://github.com/redis/redis/pull/14186)                                                | ✅     | ✅    | [link](results/result18) |
| 19 | [htop#1650](https://github.com/htop-dev/htop/issues/1650)            | [htop#1683](https://github.com/htop-dev/htop/pull/1683)                                                  | ❌     | ❌    | [link](results/result19) |
| 20 | [mbedtls#8687](https://github.com/Mbed-TLS/mbedtls/issues/8687)      | [mbedtls#8688](https://github.com/Mbed-TLS/mbedtls/pull/8688/files)                                 | ✅     | ✅    | [link](results/result20) |

## Examples

We discuss two examples in our [Website](http://172.237.148.235/). One is with stack trace in the bug report. One is without stack trace in the bug report. You can use the ID to retrieve the hypothesis as well.

### Result4 with stack trace — ID: `7a3ed78200857e0`

In this example, we included the stack trace in the bug reports.  We found that Cmind is able to refer to the similar functions for bug localization. More specifically, the issue is that ``mod`` in ApplicationAudioCaptureToolbar::Init becomes NULL when it is in the linux system and there no safety check on whether ``mod`` is NULL. However, We found that Cmind can refer other similar functions such as ``DisplayCaptureToolbar`` to localize the bugs.

---
```
Here’s a concise summary of the bug chain based solely on the provided information:

1. The helper function
   bool obs_module_get_locale_string(const obs_module_t *mod, …)
   assumes its first argument (`mod`) is non-NULL and does
   `mod->get_string(…)` without ever checking `mod` itself.

2. obs_module_get_locale_text calls obs_module_get_locale_string without adding any null-check around its `mod` parameter.

3. In all other toolbars (DisplayCaptureToolbar, AudioCaptureToolbar, WindowCaptureToolbar), after fetching a module via get_os_module they explicitly do
   if (!mod) return;
   thus avoiding any null dereference.

4. ApplicationAudioCaptureToolbar::Init, however, calls
   obs_get_module("win-wasapi") → mod
   then immediately calls
   obs_module_get_locale_text(mod, …)
   without checking whether `mod` is NULL.

5. On Linux, obs_get_module("win-wasapi") always returns NULL, so ApplicationAudioCaptureToolbar::Init ends up passing a NULL `mod` into obs_module_get_locale_text, which in turn passes it into obs_module_get_locale_string and dereferences mod->get_string, causing the crash.

Hypothesis
ApplicationAudioCaptureToolbar::Init fails to guard against a NULL module before requesting localized text, leading to a NULL-pointer dereference inside obs_module_get_locale_string. Specifically, we found that Cmind cannot refer to similar functions for bug localization in this case. 
```

### Result4 without stack trace — ID: `755c0391819b8ce`

In this example, we did not included the stack trace in the bug reports. We found that Cmind only provided a high-level hypothesis. 

---
```
Here’s what the three analyses agree on, distilled down to the two root bugs you can see in the provided reasoning:

1. Windows-only WASAPI toolbar callback is being registered on all platforms  
   – In obs_properties_dialog_register_toolbar (called as part of setting up a source’s Properties dialog), the WASAPI-specific toolbar-load callback obs_wasapi_source_toolbar_load is unconditionally hooked up—even on Linux, where it makes no sense.  
   – That means on Linux user actions end up pulling in an invalid or null function pointer.

2. Callback invocation isn’t protected against a null/invalid pointer  
   – When the toolbar-load event fires, obs_source_call_toolbar_callback (and/or obs_toolbar_load) fetches whatever callback pointer was registered and invokes it without checking for NULL or platform validity.  
   – The result on Linux is a straight null-pointer dereference in obs_wasapi_source_toolbar_load and a crash.
```


## Citation
If you use this work in an academic paper, please cite the following:
```
@article{Su2025CMIND,
  title     = {CMind: An AI Agent for Localizing C Memory Bugs},
  author    = {Chia-Yi Su and Collin McMillan},
  journal   = {xxx},
  volume    = {xxx},
  pages     = {xxx},
  year      = {2025}
}
```
