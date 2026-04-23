---
layout: leetcode-entry
title: "491. Non-decreasing Subsequences"
permalink: "/leetcode/problem/2023-01-20-491-non-decreasing-subsequences/"
leetcode_ui: true
entry_slug: "2023-01-20-491-non-decreasing-subsequences"
---

[491. Non-decreasing Subsequences](https://leetcode.com/problems/non-decreasing-subsequences/description/) medium

[https://t.me/leetcode_daily_unstoppable/91](https://t.me/leetcode_daily_unstoppable/91)

[blog post](https://leetcode.com/problems/non-decreasing-subsequences/solutions/3075577/kotlin-backtraking-set/)

```kotlin
    fun findSubsequences(nums: IntArray): List<List<Int>> {
        val res = mutableSetOf<List<Int>>()
        fun dfs(pos: Int, currList: MutableList<Int>) {
            if (currList.size > 1) res += currList.toList()
            if (pos == nums.size) return
            val currNum = nums[pos]
            //not add
            dfs(pos + 1, currList)
            //to add
            if (currList.isEmpty() || currList.last()!! <= currNum) {
                currList += currNum
                dfs(pos + 1, currList)
                currList.removeAt(currList.lastIndex)
            }
        }
        dfs(0, mutableListOf())
        return res.toList()
    }

```

Notice the size of the problem, we can do a brute force search for all solutions. Also, we only need to store the unique results, so we can store them in a set.

* we can reuse pre-filled list and do backtracking on the return from the DFS.

Space: O(2^N) to store the result, Time: O(2^N) for each value we have two choices, and we can build a binary tree of choices with the 2^n number of elements.

