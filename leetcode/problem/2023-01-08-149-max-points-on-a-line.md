---
layout: leetcode-entry
title: "149. Max Points on a Line"
permalink: "/leetcode/problem/2023-01-08-149-max-points-on-a-line/"
leetcode_ui: true
entry_slug: "2023-01-08-149-max-points-on-a-line"
---

[149. Max Points on a Line](https://leetcode.com/problems/max-points-on-a-line/) hard

[https://t.me/leetcode_daily_unstoppable/79](https://t.me/leetcode_daily_unstoppable/79)

[blog post](https://leetcode.com/problems/max-points-on-a-line/solutions/3018971/kotlin-linear-algebra-n-2/)

```kotlin
    fun maxPoints(points: Array<IntArray>): Int {
        if (points.size == 1) return 1
        val pointsByTan = mutableMapOf<Pair<Double, Double>, HashSet<Int>>()
        fun gcd(a: Int, b: Int): Int {
            return if (b == 0) a else gcd(b, a%b)
        }
        for (p1Ind in points.indices) {
            val p1 = points[p1Ind]
            for (p2Ind in (p1Ind+1)..points.lastIndex) {
                val p2 = points[p2Ind]
                val x1 = p1[0]
                val x2 = p2[0]
                val y1 = p1[1]
                val y2 = p2[1]
                var dy = y2 - y1
                var dx = x2 - x1
                val greatestCommonDivider = gcd(dx, dy)
                dy /= greatestCommonDivider
                dx /= greatestCommonDivider
                val tan = dy/dx.toDouble()
                val b = if (dx == 0) x1.toDouble() else (x2*y1 - x1*y2 )/(x2-x1).toDouble()
                val line = pointsByTan.getOrPut(tan to b, { HashSet() })
                line.add(p1Ind)
                line.add(p2Ind)
            }
        }
        return pointsByTan.values.maxBy { it.size }?.size?:0
    }

```

Just do the linear algebra to find all the lines through each pair of points.
Store `slope` and `b` coeff in the hashmap. Also, compute `gcd` to find precise slope. In this case it works for `double` precision slope, but for bigger numbers we need to store `dy` and `dx` separately in `Int` precision.

Space: O(n^2), Time: O(n^2)

