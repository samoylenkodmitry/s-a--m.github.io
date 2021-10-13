---
layout: post
title: Databinding Episode II; Past Mistakes
---
# Databinding Эпизод II; Ошибки прошлого

Иногда в повседневной разработке под Android сталкиваешься со странными, если не сказать мистическими багами. (стандартное вступление)
В предыдущем эпизоде http://dmitrysamoylenko.com/2019/04/16/databinding_hidden_danger.html

Предположим, имеется класс *.kt:

```
Base.kt
open class Base {
  open fun isEmpty() = false
}
```

и его наследник *.java:

```
Child.java
class Child extends Base {
  public boolean isEmpty = true;
}
```

Уже тут можно заметить, что мы выстрелили себе в ногу. Но продолжим.
Мы захотели использовать класс Child в databidning-е. 

Внимание, вопрос: будет ли видна следующая View?

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

# Мораль истории
1. Не наследуйте java-классы от kotlin-классов
2. А может быть пора переходить на Jetpack Compose?
