---
layout: post
title: First Look at Jetpack Compose
---
# Let's Create a Hundred Thousand Views
I should say upfront that this article is not intended to nitpick this library, as I personally find it quite appealing. 
First, we'll create a simple list to add views to. 
This is an inefficient method, but it will illustrate the overall picture of how much attention was paid to performance in the new framework.
So, let's compare the old method:
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
And the new method:
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
This is a simple scrolling list, where N text views are arranged vertically.

We measure performance using Android Studio's `Profiler` tab after the heap growth stabilizes.
1. N = 1
![c_1]({{ site.url }}/assets/c_1.png)
Heap size for View - 1.5M vs Compose - 2.7M
This is the basic difference with just one view. Twice as much, but not critical for modern devices.

2. N = 40,000
![c_2]({{ site.url }}/assets/c_2.png)
View 723M vs Compose 26778M
You can see how memory usage significantly increases depending on the number of elements in compose.

3. N = 60,000 here the emulator ran out of heap space of 512 MB (with allocated RAM=30GB) and compose crashed with OutOfMemoryError

4. N = 100,000 continue testing View - 1807M. There's a huge potential for growth in the number of elements present at the same time.
![c_3]({{ site.url }}/assets/c_3.png)

Let's plot the memory growth against the number of elements.
![c_4]({{ site.url }}/assets/c_4.png)
Memory usage for View grows linearly, which is not the case for Compose.

# The Real Power of Jetpack Compose

It takes just 3 lines to create a lazy list like RecyclerView

```
LazyColumnFor(items = (1..1_000_000).toList()) {
  Text(text="Hello $it")
}
```
For comparison, achieving the same result using RecyclerView:

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
Let's take a closer look at the Lazy list. 
It's interesting to see how often it redraws elements. 
Let's draw multicolored circles using Canvas:

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

As we can see, Canvas() redraws every tick
![compose_lazy_flashes.gif]({{ site.url }}/assets/compose_lazy_flashes.gif)

However, Text() redraws only when it leaves the visible area:
![compose_lazy_text_redraw.gif]({{ site.url }}/assets/compose_lazy_text_redraw.gif)

This is encouraging: they are thinking about component redraw optimizations in advance.

Also, the `LazyColumnFor(items)` function interface does not yet allow creating a truly infinite list. A finite set of items is always expected.

# tl;dr
Jetpack Compose is still in alpha version and it's evident that priority is given to API conciseness. Let's hope that Google doesn't stop at the first iteration and optimizes the components. 

After all, it's their own slogan - #pert_matters, which they gradually seem to forget <https://www.youtube.com/playlist?list=PLWz5rJ2EKKc9CBxr3BVjPTPoDPLdPIFCE>)

<iframe width="560" height="315" src="https://www.youtube.com/embed/videoseries?list=PLWz5rJ2EKKc9CBxr3BVjPTPoDPLdPIFCE" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

