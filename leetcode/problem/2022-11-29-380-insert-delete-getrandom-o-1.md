---
layout: leetcode-entry
title: "380. Insert Delete GetRandom O(1)"
permalink: "/leetcode/problem/2022-11-29-380-insert-delete-getrandom-o-1/"
leetcode_ui: true
entry_slug: "2022-11-29-380-insert-delete-getrandom-o-1"
---

[380. Insert Delete GetRandom O(1)](https://leetcode.com/problems/insert-delete-getrandom-o1/) medium

[https://t.me/leetcode_daily_unstoppable/35](https://t.me/leetcode_daily_unstoppable/35)

```kotlin

class RandomizedSet() {
    val rnd = Random(0)
    val list = mutableListOf<Int>()
    val vToInd = mutableMapOf<Int, Int?>()
    fun insert(v: Int): Boolean {
        if (!vToInd.contains(v)) {
            vToInd[v] = list.size
            list.add(v)
            return true
        }
        return false
    }
    fun remove(v: Int): Boolean {
        val ind = vToInd[v] ?: return false
        val prevLast = list[list.lastIndex]
        list[ind] = prevLast
        vToInd[prevLast] = ind
        list.removeAt(list.lastIndex)
        vToInd.remove(v)
        return true
    }
    fun getRandom(): Int = list[rnd.nextInt(list.size)]
}

```

The task is simple, one trick is to remove elements from the end of the list, and replacing item with the last one.
Some thoughts:
* don't optimize lines of code, that can backfire. You can use syntax sugar, clever operations inlining, but also can shoot in the foot.

O(1) time, O(N) space

