---
layout: leetcode-entry
title: "944. Delete Columns to Make Sorted"
permalink: "/leetcode/problem/2023-01-03-944-delete-columns-to-make-sorted/"
leetcode_ui: true
entry_slug: "2023-01-03-944-delete-columns-to-make-sorted"
---

[944. Delete Columns to Make Sorted](https://leetcode.com/problems/delete-columns-to-make-sorted/description/) easy

[https://t.me/leetcode_daily_unstoppable/73](https://t.me/leetcode_daily_unstoppable/73)

[blog post](https://leetcode.com/problems/delete-columns-to-make-sorted/solutions/2992229/kotlin-do-what-is-asked/)

```kotlin
    fun minDeletionSize(strs: Array<String>): Int =
       (0..strs[0].lastIndex).asSequence().count { col ->
           (1..strs.lastIndex).asSequence().any { strs[it][col] < strs[it-1][col] }
        }

```

Just do what is asked.
We can use Kotlin's `sequence` api.

Space: O(1), Time: O(wN)

