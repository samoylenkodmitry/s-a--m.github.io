---
layout: leetcode-entry
title: "1187. Make Array Strictly Increasing"
permalink: "/leetcode/problem/2023-06-17-1187-make-array-strictly-increasing/"
leetcode_ui: true
entry_slug: "2023-06-17-1187-make-array-strictly-increasing"
---

[1187. Make Array Strictly Increasing](https://leetcode.com/problems/make-array-strictly-increasing/description/) hard
[blog post](https://leetcode.com/problems/make-array-strictly-increasing/solutions/3647345/kotlin-dfs-memo/)
[substack](https://dmitriisamoilenko.substack.com/p/17062023-1187-make-array-strictly?sd=pf)

![image.png](/assets/leetcode_daily_images/a5076f19.webp)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/248
#### Problem TLDR
Minimum replacements to make `arr1` increasing using any numbers `arr2`
#### Intuition
For any current position in `arr1` we can leave this number or replace it with any number from `arr2[i] > curr`. We can write Depth-First Search to check all possible replacements. To memorize, we must also consider the previous value. It can be used as-is, but more optimally, we just store a `skipped` boolean flag and restore the `prev` value: if it was skipped, then previous is from `arr1` else from `arr2`.

#### Approach
* sort and distinct the `arr2`
* use `Array` for cache, as it will be faster than a `HashMap`
* use explicit variable for the invalid result
* for the stop condition, if all the `arr1` passed, then result it good
#### Complexity
- Time complexity:
$$O(n^2)$$
- Space complexity:
$$O(n^2)$$

#### Code

```kotlin

fun makeArrayIncreasing(arr1: IntArray, arr2: IntArray): Int {
    val list2 = arr2.distinct().sorted()
    val INV = -1
    val cache = Array(arr1.size + 1) { Array(list2.size + 1) { IntArray(2) { -2 } } }
    fun dfs(pos1: Int, pos2: Int, skipped: Int): Int {
        val prev = if (skipped == 1) arr1.getOrNull(pos1-1)?:-1 else list2.getOrNull(pos2-1)?:-1
        return if (pos1 == arr1.size) 0 else cache[pos1][pos2][skipped].takeIf { it != -2} ?:
        if (pos2 == list2.size) {
            if (arr1[pos1] > prev) dfs(pos1 + 1, pos2, 1) else INV
        } else if (list2[pos2] <= prev) {
            dfs(pos1, pos2 + 1, 1)
        } else {
            val replace = dfs(pos1 + 1, pos2 + 1, 0)
            val skip = if (arr1[pos1] > prev) dfs(pos1 + 1, pos2, 1) else INV
            if (skip != INV && replace != INV) minOf(skip, 1 + replace)
            else if (replace != INV) 1 + replace else skip
        }.also { cache[pos1][pos2][skipped] = it }
    }
    return dfs(0, 0, 1)
}

```

