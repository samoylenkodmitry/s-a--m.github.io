---
layout: post
title: Databinding Episode II; Past Mistakes
---

# Databinding Episode II; Mistakes of the Past

Sometimes in everyday Android development, you encounter strange, if not mystical bugs. (standard introduction)

In the previous episode [Databinding Episode I; Hidden Danger](http://dmitrysamoylenko.com/2019/04/16/databinding_hidden_danger.html)

Let's assume there is a \*.kt class:

```
Base.kt
open class Base {
  open fun isEmpty() = false
}
```

and its \*.java inheritor:

```
Child.java
class Child extends Base {
  public boolean isEmpty = true;
}
```

Here we can already notice that we have shot ourselves in the foot. But let's continue.

We wanted to use the Child class in databinding. 

Attention, question: will the following View be visible?

```
some.xml
<layout>
<data>
  <variable name="child" type="Child" />
</data>

<View
  ...
  android:visibility="@{child.isEmpty ? View.VISIBLE : View.GONE}"
  ...
```

# Moral of the Story

1. Do not inherit Java classes from Kotlin classes.
2. Maybe it's time to switch to Jetpack Compose?
