---
layout: leetcode-entry
title: "1416. Restore The Array"
permalink: "/leetcode/problem/2023-04-23-1416-restore-the-array/"
leetcode_ui: true
entry_slug: "2023-04-23-1416-restore-the-array"
---

[1416. Restore The Array](https://leetcode.com/problems/restore-the-array/description/) hard

```kotlin

fun numberOfArrays(s: String, k: Int): Int {
    // 131,7  k=1000
    // 1317 > 1000
    // 20001  k=2000
    // 2      count=1
    //  000   count=1, curr=2000
    //     1  count++, curr=1
    //
    // 220001 k=2000
    // 2      count=1 curr=1
    // 22     count+1=2 curr=22          [2, 2], [22]
    // 220    curr=220                   [2, 20], [220]
    // 2200   curr=2200 > 2000, curr=200 [2, 200], [2200]
    // 22000  curr=2000   count=1        [2, 2000]
    // 220001 count+1=3 curr=20001 > 2000, curr=1  [2, 2000, 1], []
    val m = 1_000_000_007L
    val cache = LongArray(s.length) { -1L }
    fun dfs(curr: Int): Long {
        if (curr == s.length) return 1L
        if (s[curr] == '0') return 0L
        if (cache[curr] != -1L) return cache[curr]
        var count = 0L
        var num = 0L
        for (i in curr..s.lastIndex) {
            val d = s[i].toLong() - '0'.toLong()
            num = num * 10L + d
            if (num > k) break
            val countOther = dfs(i + 1)
            count = (count + countOther) % m
        }
        cache[curr] = count
        return count
    }
    return dfs(0).toInt()
}

or bottom-up

fun numberOfArrays(s: String, k: Int): Int {
    val cache = LongArray(s.length)
    for (curr in s.lastIndex downTo 0) {
        if (s[curr] == '0') continue
        var count = 0L
        var num = 0L
        for (i in curr..s.lastIndex) {
            num = num * 10L + s[i].toLong() - '0'.toLong()
            if (num > k) break
            val next = if (i == s.lastIndex) 1 else cache[i + 1]
            count = (count + next) % 1_000_000_007L
        }
        cache[curr] = count
    }
    return cache[0].toInt()
}

memory optimization:

fun numberOfArrays(s: String, k: Int): Int {
    val cache = LongArray(k.toString().length + 1)
    for (curr in s.lastIndex downTo 0) {
        System.arraycopy(cache, 0, cache, 1, cache.size - 1)
        if (s[curr] == '0') {
            cache[0] = 0
            continue
        }

        var count = 0L
        var num = 0L
        for (i in curr..s.lastIndex) {
            num = num * 10L + s[i].toLong() - '0'.toLong()
            if (num > k) break
            val next = if (i == s.lastIndex) 1 else cache[i - curr + 1]
            count = (count + next) % 1_000_000_007L
        }
        cache[0] = count
    }
    return cache[0].toInt()
}

```

[blog post](https://leetcode.com/problems/restore-the-array/solutions/3446057/kotlin-choose-dp-rule/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-23042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/189
#### Intuition
One naive solution, is to find all the possible ways of splitting the string, and calculating `soFar` number, gives TLE as we must take `soFar` into consideration when memoizing the result.
Let's consider, that for every position in `s` there is only one number of possible arrays. Given that, we can start from each position and try to take the `first` number in all possible correct ways, such that `num < k`. Now, we can cache this result for reuse.

#### Approach
* use `Long` to avoid overflow
* we actually not need all the numbers in cache, just the $$lg(k)$$ for the max length of the number
#### Complexity
- Time complexity:
$$O(nlg(k))$$
- Space complexity:
$$O(lg(k))$$

