---
layout: post
title: Делаем Blur View в Android
---
# Делаем Blur View в Android

Сперва рассмотрим готовые варианты. Самые популярные библиотеки

1. https://github.com/Dimezis/BlurView
+ простая
- реализована в виде готовой View, от которой нужно наследоваться
- все операции выполняются в главном потоке
2. https://github.com/wasabeef/Blurry
+ красивое апи
+ поддерживает асинхронное выполнение
- неэффективный захват изображения View (через drawingCache)
- не кеширует и не переиспользует Bitmap

Также есть отличный подробный обзор в серии статей:
https://blog.stylingandroid.com/blurring-images-part-1/ 

Итак, что нужно учесть, при реализации блюра:
0. Так как блюр динамический, то его перерисовка может вызываться в каждый тик рендера.
1. Нужно не просто уложиться в 15мс (60 кадров/секунду), а сделать это намного быстрее. В Android при либой анимации view происходит
следующее:
+ вызов layout-measure для всех view в иерархии на экране
+ вызов onDraw для всех view
+ вызов изменений параметров для анимированной view базируясь на времени анимации
+ вернуться ко второму шагу
Так как мы не хотим одной красивой заблюренной вьюшкой затормаживать весь интерфейс и все анимации, то времени мы должны занимать
по-минимому.
2. Нужно не создавать лишних объектов. 
Сборщику мусора это не понравится. Тут действует то же правило, что и при реализации метода onDraw во View https://developer.android.com/training/custom-views/custom-drawing.
3. API нашего компонента не должно вызывать головную боль у программиста. Пользователь апи не должен заниматься расчетом взаимных
позиций view. Достаточно, если у него будет возможность указать, когда _точно_ нужно перерисовать блюр. 
Либо же вообще ничего не указывать, тогда компонент должен подписаться на ViewTreeObserver.OnPreDrawListener.

## Исполнение

### api:
```java
public class Blurer {
	public void start(final View image, final View layout) {}
	public void stop() {}
}
```

Плюс, добавим возможность перерисовки при скролле (слушатель будет планировать задачу на перерисовку):
```java
	public RecyclerView.OnScrollListener getScrollListener() {}
```

### Последовательность действий
1. Расчитать размеры
2. "Сфотографировать" view
3. Применить blur
4. Установить фон

//TODO implementation
//TODO benchmark

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |