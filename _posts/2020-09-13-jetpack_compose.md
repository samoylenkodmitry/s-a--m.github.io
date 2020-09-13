---
layout: post
title: Первый взгляд на Jetpack Compose
---
# Создадим сто тысяч вьюх
Сразу скажу, что данная статья не ставит задачу придраться к этой библиотеке, т.к. мне самому она очень симпатична. 
Сначала создадим простой список, куда будем добавлять вьюхи. 
Это неоптимальный способ, но он покажет общую картинку, насколько в новом фреймворке уделялось внимание производительности.
Итак, сравним старый способ:
```
class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val N = 100_000
        setContentView(
            ScrollView(this).apply {
                addView(
                    LinearLayout(context).apply {
                        orientation = LinearLayout.VERTICAL
                        repeat(N) {
                            addView(
                                TextView(context)
                                    .apply { text = "hello $it" })
                        }
                    })
            })
    }
}
```
И новый способ:
```
class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val N = 100_000
        setContent {
            MyApplicationTheme {
                Surface(color = MaterialTheme.colors.background) {
                    ScrollableColumn {
                        repeat(N) {
                            Text(text = "Hello $it")
                        }
                    }
                }
            }
        }
    }
}
```
Это простой скроллящийся список, в котором N текстовых вьюх расположены по вертикали.

Показатели замеряем с помощью Android Studio на вкладке `Profiler` после некоторого устаканивания роста хипа.
1. N = 1

![c_1]({{ site.url }}/assets/c_1.png)
Размер heap View - 1.5M vs размер Compose - 2.7M
Это базовое различие при всего лишь одной вьюхе. В 2 раза, но не критично для современных девайсов.

2. N = 40_000
![c_2]({{ site.url }}/assets/c_2.png)
View 723M vs Compose 26778M
Видно насколько сильно растет память в зависимости от количества элементов у compose.

3. N = 60_000 тут у эмулятора не хватило heap в размере 512 мб (при выделенной RAM=30GB) и compose упал с OutOfMemoryError

4. N = 100_000 продолжаю тестировать View - 1807М. Очень большой потенциал роста количества одновременно присутствующих элементов.
![c_3]({{ site.url }}/assets/c_3.png)

Составим график роста памяти от количества элементов.
![c_4]({{ site.url }}/assets/c_4.png)
Объем памяти View растет линейно, чего не скажешь про Compose. Штошш.

# Настоящая сила jetpack compose

Нужно всего 3 строки для создания lazy-списка типа RecyclerView

```
LazyColumnFor(items = (1..1_000_000).toList()) {
  Text(text="Hello $it")
}
```
Для сравнения, тот же результат с помощью RecyclerView:
```
RecyclerView(this).apply {
  adapter = object : RecyclerView.Adapter<RecyclerView.ViewHolder>() {
     override fun onCreateViewHolder(parent: ViewGroup, viewType: Int) =
        object : RecyclerView.ViewHolder(TextView(context)) {}

     override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        (holder.itemView as TextView).text = "hello $position"
     }

     override fun getItemCount() = N

  }
  layoutManager = LinearLayoutManager(context)
}

```

Рассмотрим Lazy-список подробнее. 
Интересно, как часто он перерисовывает элементы. 
Нарисуем с помощью Canvas разноцветные круги:

```
LazyColumnFor(items = (1..N).toList()) {
  Text(text = "Hello $it")
  Canvas(modifier = Modifier.size(10.dp), onDraw = {
     this.drawOval(
        color = Color(
           Math.random().toFloat(),
           Math.random().toFloat(),
           Math.random().toFloat()
        ), size = this.size.times(3.0f)
     )
  })
}

```

Как видим, Canvas перерисовывается каждый тик
![compose_lazy_flashes.gif]({{ site.url }}/assets/compose_lazy_flashes.gif)

Однако TextView перерисовывается только при выходе за границу видимой области:
![compose_lazy_text_redraw.gif]({{ site.url }}/assets/compose_lazy_text_redraw.gif)
Это радует: об оптимизациях прорисовки компонентов думают заранее.

Также интерфейс функции `LazyColumnFor(items) ` пока что не позволяет создать по-настоящему бесконечный список. Всегда ожидается конечное множество items элементов.

# TLDR;
Jetpack Compose пока что находится в альфа версии и видно, что приоритет отдается лаконичности API. Будет надеяться, что гугл не остановится на первой итерации и оптимизирует компоненты. 

В конце концов, это их собственный лозунг #pert_matters https://developer.android.com/topic/performance 
Про который они постепенно забывают (все помнят, что енумы и автобоксинг - это плохо? :) https://www.youtube.com/playlist?list=PLWz5rJ2EKKc9CBxr3BVjPTPoDPLdPIFCE )

