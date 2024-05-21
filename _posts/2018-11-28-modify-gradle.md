---
layout: post
title: Modifying Gradle Options on the Fly
---

# Modifying Gradle Options on the Fly

Often, Google lets us try out their unfinished developments. For example, the new R8 resource compiler.
It has replaced the "old" Proguard, is faster, and now also handles class removal. 
You can enable it in gradle.properties:
```
android.enableR8=true
```

And if you have something like this in your build.gradle:

```
	buildTypes {

		release {
...
			minifyEnabled true
...
		}
  }
```
You can only learn about its stability by building a release APK. If you don't do this often, problems are inevitable.
For instance, recently it removed part of the fields, annotations, and classes in our project. It did so neatly,
that we could only detect it when one of the pages of the application failed to load.

We wanted fast assembly, so the task arose to allow R8 only in builds from the studio.

So, here's how it can be done through build.gradle:

```
boolean isInvokeFromIde = project.properties['android.injected.invoked.from.ide']//определяем сборку из студии
this['android.enableR8']  = isInvokeFromIde//меняем настройку gradle.properties в build.gradle в рантайме
```
The substitution must occur before the android-plugin is called.






