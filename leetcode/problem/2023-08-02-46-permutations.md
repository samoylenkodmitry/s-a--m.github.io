---
layout: leetcode-entry
title: "46. Permutations"
permalink: "/leetcode/problem/2023-08-02-46-permutations/"
leetcode_ui: true
entry_slug: "2023-08-02-46-permutations"
---

[46. Permutations](https://leetcode.com/problems/permutations/description/) medium
[blog post](https://leetcode.com/problems/permutations/solutions/3850880/kotlin-dfs-backtrack-bitmask/)
[substack](https://dmitriisamoilenko.substack.com/p/02082023-46-permutations?sd=pf)
![image.png](/assets/leetcode_daily_images/0bc84ba5.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/296

#### Problem TLDR

List of all numbers permutations

#### Intuition

As the total count of number is small, we can just brute force the solution. We can use DFS and a backtracking technique: add number to the list pre-order then remove it post-order.

#### Approach

Iterate over all numbers and choose every number not in a `bit mask`

#### Complexity

- Time complexity:
$$O(n * n!)$$, as we go `n * (n - 1) * (n - 2) * .. * 2 * 1`

- Space complexity:
$$(n!)$$

#### Code

```kotlin

    fun permute(nums: IntArray): List<List<Int>> = mutableListOf<List<Int>>().apply {
      val list = mutableListOf<Int>()
      fun dfs(mask: Int): Unit = if (list.size == nums.size) this += list.toList()
        else nums.forEachIndexed { i, n ->
          if (mask and (1 shl i) == 0) {
            list += n
            dfs(mask or (1 shl i))
            list.removeAt(list.lastIndex)
          }
        }
      dfs(0)
    }

```

