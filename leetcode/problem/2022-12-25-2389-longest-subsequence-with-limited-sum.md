---
layout: leetcode-entry
title: "2389. Longest Subsequence With Limited Sum"
permalink: "/leetcode/problem/2022-12-25-2389-longest-subsequence-with-limited-sum/"
leetcode_ui: true
entry_slug: "2022-12-25-2389-longest-subsequence-with-limited-sum"
---

[2389. Longest Subsequence With Limited Sum](https://leetcode.com/problems/longest-subsequence-with-limited-sum/description/) easy

[https://t.me/leetcode_daily_unstoppable/63](https://t.me/leetcode_daily_unstoppable/63)

[blog post](https://leetcode.com/problems/longest-subsequence-with-limited-sum/solutions/2948494/kotlin-sort-prefix-sum-binary-search/)

```kotlin
    fun answerQueries(nums: IntArray, queries: IntArray): IntArray {
       nums.sort()
       for (i in 1..nums.lastIndex) nums[i] += nums[i-1]
       return IntArray(queries.size) {
           val ind = nums.binarySearch(queries[it])
           if (ind < 0) -ind-1 else ind+1
       }
    }

```

We can logically deduce that for the maximum number of arguments we need to take as much as possible items from the smallest to the largest.
We can sort items. Then pre-compute `sums[i] = sum from [0..i]`. Then use binary search target sum in sums. Also, can modify `nums` but that's may be not necessary.

Space: O(N), Time: O(NlogN)

