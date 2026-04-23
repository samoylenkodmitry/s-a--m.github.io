---
layout: leetcode-entry
title: "3625. Count Number of Trapezoids II"
permalink: "/leetcode/problem/2025-12-03-3625-count-number-of-trapezoids-ii/"
leetcode_ui: true
entry_slug: "2025-12-03-3625-count-number-of-trapezoids-ii"
---

[3625. Count Number of Trapezoids II](https://leetcode.com/problems/count-number-of-trapezoids-ii) hard
[blog post](https://leetcode.com/problems/count-number-of-trapezoids-ii/solutions/7389188/kotlin-by-samoylenkodmitry-qcg4/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/03112025-3625-count-number-of-trapezoids?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Ka-m6HHekxg)

![4936712f-8071-4c9b-aa99-2728efa40b4e (1).webp](/assets/leetcode_daily_images/5454fc60.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1192

#### Problem TLDR

Count trapezoids from points #hard #geometry

#### Intuition

```j
    // i really don't like this problem
    //
    // line y = (dy/dx)x + b
    //      y*(x2-x1) = x*(y2-y1)
    //
    // how to track parallel lines?
    // we have only 500 points, can be O(n^2)
    //
    // for each point pair: count same slope others
    //

    // the key should be the line, not the slope
    //
    // (y-y0)= (x-x0)*(y2-y1)/(x2-x1), slope + point of intersection of x and y coordinates
    // line intersect X coordinate at x = x0, y = 0, Y coordinate at y0
    //
    // i don't have a time to remember geometry
    //
    // how to check y-intercept?; y = kx+b; b = y -kx
    //
    // ok the double is not precise enough, have false positive match
    // -24,-89  42,11      and     -75,-89  -9,11     (or maybe it is symmetrical)
    //
    // let's go for answer, feels pointless, how to use gcd here?
```
* key slopes by normalized dy/gcd(dx,dy) | dx/gcd(dx,dy)
* key parallelogram by equal distances between points (they are pairly-equal for parallelogram) dy | dx
* then count subprolem with points on parallel lines: sum+count, res += sum*count

#### Approach

* parallelogram's keyed points gives exactly 2x for parallelograms and 1x otherwise, so /2 would gives only duplicates

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin
// 578ms
    fun countTrapezoids(p: Array<IntArray>): Int {
        val m = HashMap<Int, HashMap<Int, Int>>(); val d = m.toMutableMap()
        fun gcd(a: Int, b: Int): Int = if (b == 0) a else gcd(b, a % b)
        for (i in p.indices) for (j in i+1..<p.size) {
            val (x1,y1) = p[i]; val (x2,y2) = p[j]; var dx = x2-x1; var dy = y2-y1
            if (dx < 0 || dx == 0 && dy < 0) { dx = -dx; dy = -dy }
            val g = gcd(dx, dy); val sx = dx / g; val sy = dy / g; val line = sx*y1-sy*x1
            val slope = (sx shl 12) or (sy+2000); val dist = (dx shl 12) or (dy+2000)
            m.getOrPut(slope) { HashMap() }.merge(line, 1, Int::plus)
            d.getOrPut(dist) { HashMap() }.merge(line, 1, Int::plus)
        }
        fun cnt(m: Map<Int, Map<Int, Int>>): Int =  m.values.sumOf {
            it.values.fold(0 to 0) { (sum, r), t -> (sum + t) to (r + sum*t)}.second }
        return cnt(m) - cnt(d)/2
    }
```

