---
layout: post
title: A Cool Refactoring Hidden in IntelliJ IDEA
---
# This is about replacing field access with a getter
By default, there's a visible refactoring option for this through `alt+insert` - `getter`. 

![enq1]({{ site.url }}/assets/enq1.png)

The problem is that if a field is already used in many places, just creating a getter is not enough. You also need to replace direct access with this getter everywhere.

![enq2]({{ site.url }}/assets/enq2.png)

Fortunately, JetBrains has thought of this! 
Press `ctrl+shift+A`, enter and open the `refactor this` action.

![enq3]({{ site.url }}/assets/enq3.png)

In the menu, there are many different refactorings. We select `encapsulate`.

![enq4]({{ site.url }}/assets/enq4.png)

A dialog will open where you select the needed fields, tick the getters/setters, and click `refactor`.

![enq5]({{ site.url }}/assets/enq5.png)

Done! 

![enq6]({{ site.url }}/assets/enq6.png)

As we can see, direct access has been replaced with the getter throughout the code.

![enq7]({{ site.url }}/assets/enq7.png)
