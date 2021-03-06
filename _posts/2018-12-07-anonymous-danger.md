---
layout: post
title: Чем опасны анонимные классы
---
# Никогда не недооценивай опасность повседневных вещей

Иногда в повседневной разработке под Android сталкиваешься со странными, если не сказать мистическими багами.
Дает о себе знать и фрагментарность платформы и различия в самих устройствах.

В этот раз я делал анимированный Drawable. В механизме обновления Drawable участвует Callback, внутри которого нужно сообщить
View о необходимости перерисовки.
```
//MyView.java

new Drawable.Callback() {
 @Override
 public void invalidateDrawable(@NonNull final Drawable who) {
  invalidate();// <-- тут вызывается invalidate для View, в которой находится Drawable
 }
		
...
}
```

Внутри Drawable я вызываю на каждом тике анимации метод Drawable#invalidateSelf(), который и дергает этот колбэк.

```
//MyDrawable.java

 public void startAnimation() {
  final ValueAnimator valueAnimator = ValueAnimator.ofInt(0, 100);
  valueAnimator.addUpdateListener(animation ->	{
   ...
   invalidateSelf();// <--- тут Drawable "говорит" View через Callback о том что надо бы перерисоваться
  });
  valueAnimator.start();
 }
```

![triangle]({{ site.url }}/assets/triangle.gif)

Казалось бы работа выполнена, но любопытство и опыт работы с андроидом заставили меня проверить как это 
работает на других версиях системы. И, как оказалось, сомнения были не без причины - на Android 6 анимация
работала не до конца, останавливаясь на полпути или вообще не начиналась!

![triangle_not_work]({{ site.url }}/assets/triangle_not_work.png)

В ходе долгих поисков и логгирования был найден виновник - это был Drawable. В процессе анимации он почему-то терял колбэк:

```
//MyDrawable.java
valueAnimator.addUpdateListener(animation ->	{
 ...
 invalidateSelf();
 ...
 Log.d("x", "" + getCallback());// <-- в середине анимации этот метод начинал возвращать null
});
```

Заглянув внутрь, обнаруживаю "надежный" способ избежать утечки памяти от команды Google:
```
//android.graphics.drawable.Drawable.java
 public final void setCallback(@Nullable Callback cb) {
  mCallback = cb != null ? new WeakReference<>(cb) : null; // <-- зачем заботиться об утечках, пусть GC позаботится!
 }
```

Немного погоревав, решил, что делать нечего, придется жить с тем, что имеем. В моей реализации я устанавливал колбэк в виде
лямбды:
```
//MyView.java
mSplashDrawable.setCallback(new Drawable.Callback() { // <-- анонимный класс
...
});
```
Gc работает с WeakReference подсчитывая ссылки, и если не находит ни одной жесткой ссылки, очищает его. 
Лямбда сама по себе является анонимным классом и содержит жесткую ссылку на внешний класс. Цепочка выглядит так
```
MyView -> Drawable -> WeakReference -> Callback
       <------------------------------|         // <-- ссылка анонимного класса на родителя                      
```

На объект Callback указывает только WeakReference, поэтому GC его и подобрал.

Заменил анонимный класс на поле и больше GC мой WeakReference не подчищал:
```
//MyView
 private final Drawable.Callback mCallback = new Drawable.Callback() {
  ...
 }		
```

![triangle]({{ site.url }}/assets/triangle.gif)

Вывод такой:
будьте осторожны с анонимными классами и еще осторожнее с Android SDK ༼ʘ̚ل͜ʘ̚༽
