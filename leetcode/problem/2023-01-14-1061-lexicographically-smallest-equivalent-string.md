---
layout: leetcode-entry
title: "1061. Lexicographically Smallest Equivalent String"
permalink: "/leetcode/problem/2023-01-14-1061-lexicographically-smallest-equivalent-string/"
leetcode_ui: true
entry_slug: "2023-01-14-1061-lexicographically-smallest-equivalent-string"
---

[1061. Lexicographically Smallest Equivalent String](https://leetcode.com/problems/lexicographically-smallest-equivalent-string/description/) medium

[https://t.me/leetcode_daily_unstoppable/85](https://t.me/leetcode_daily_unstoppable/85)

[blog post](https://leetcode.com/problems/lexicographically-smallest-equivalent-string/solutions/3049304/kotlin-uniton-find/)

```kotlin
    fun smallestEquivalentString(s1: String, s2: String, baseStr: String): String {
        val uf = IntArray(27) { it }
        fun find(ca: Char): Int {
            val a = ca.toInt() - 'a'.toInt()
            var x = a
            while (uf[x] != x) x = uf[x]
            uf[a] = x
            return x
        }
        fun union(a: Char, b: Char) {
            val rootA = find(a)
            val rootB = find(b)
            if (rootA != rootB) {
                val max = maxOf(rootA, rootB)
                val min = minOf(rootA, rootB)
                uf[max] = min
            }
        }
        for (i in 0..s1.lastIndex) union(s1[i], s2[i])
        return baseStr.map { (find(it) + 'a'.toInt()).toChar() }.joinToString("")
    }

```

We need to find connected groups, the best way is to use the Union-Find.

Iterate over strings and connect each of their chars.
* to find a minimum, we can select the minimum of the current root.

Space: O(N) for storing a result, Time: O(N)

