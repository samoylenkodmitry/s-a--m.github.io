---
layout: post
title: Подключаем дебаггер к процессу сборки Android-приложения
---
# Для этого нужно всего лишь...
1. В Android Studio выбрать пункт Edit Configurations
![edit_configurations]({{ site.url }}/assets/editconfigurations.png)
2. В нем добавить новую конфигурацию Remote. Менять настройки не нужно, жмем ок:
![debug_jvm]({{ site.url }}/assets/debug_jvm_ok.png)
3. Запустить сборку из командной строки:
```
 ./gradlew --no-daemon -Dorg.gradle.debug=true -Dkotlin.daemon.jvm.options="-Xdebug,-Xrunjdwp:transport=dt_socket,address=5005,server=y,suspend=y" assemble
```
4. Сразу кликнуть на ![debug]({{ site.url }}/assets/debug.png) и не забыть поставить брекпоинт в градл-скрипте.


А также это поможет отладить кастомный Annotation Processor.
