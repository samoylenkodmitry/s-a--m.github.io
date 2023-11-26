---
layout: post
title: Connecting the Debugger to the Android App Build Process
---
# To do this, you only need to...
1. In Android Studio, select the Edit Configurations item
![edit_configurations]({{ site.url }}/assets/edit_configurations.png)
2. In it, add a new Remote configuration. No need to change the settings, just click OK:
![debug_jvm]({{ site.url }}/assets/debug_jvm_ok.png)
3. Start the build from the command line:
```
 ./gradlew --no-daemon -Dorg.gradle.debug=true -Dkotlin.daemon.jvm.options="-Xdebug,-Xrunjdwp:transport=dt_socket,address=5005,server=y,suspend=y" assemble
```
4. Immediately click on ![debug]({{ site.url }}/assets/debug.png) and don't forget to set a breakpoint in the Gradle script.

![debug_breakpoint]({{ site.url }}/assets/debug_jvm_breakpoint.png)

This will also help debug a custom Annotation Processor.