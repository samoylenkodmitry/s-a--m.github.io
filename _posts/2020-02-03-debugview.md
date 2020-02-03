---
layout: post
title: Как дебажить верстку в Android
---
# Как дебажить верстку в Android

Иногда верстка начинает показывать магию: скрывать в ненужный момент контролы, неправильно определять свои размеры и положение.
Обычно такое бывает когда код, управляющий версткой, излишне сложен. Например, когда одна часть приложения пытается скрыть вьюху, другая показать. 
И выполняются эти действия в произвольные моменты времени.

Как правило, это говорит о неправильной архитектуре, и правильным решением было бы переписать сложное управление на модель redux с одним источником правды.

Но если у вас нет времени переписывать большую часть приложения, поможет следующий подход:
* создаем класс наследующий исследуемую view  (в данном примере FrameLayout) и переопределяем метод setVisibility.
```java
class DebugView @JvmOverloads constructor(
	context: Context, attrs: AttributeSet? = null, defStyleAttr: Int = 0
) : FrameLayout(context, attrs, defStyleAttr) {
	init {
		if (!BuildConfig.DEBUG) {
			Assert.fail("Don't forget to remove debug view from release app")
		}
	}

	override fun setVisibility(visibility: Int) {
		super.setVisibility(visibility)

		// this is example of how to use it for listening to view changes
		Error("debug! $visibility $tag ${getTag(R.id.adjust_height)}").printStackTrace()
	}
}
```
в верстке заменяем исследуемую view на нашу DebugView:

```xml
	<DebugView
		android:layout_width="match_parent"
		android:layout_height="wrap_content"
		>

		<TextView
...
			  
```

Теперь, когда в коде кто-либо поменяет видимость исследуемой view, то это можно будет поймать.
Готово!
