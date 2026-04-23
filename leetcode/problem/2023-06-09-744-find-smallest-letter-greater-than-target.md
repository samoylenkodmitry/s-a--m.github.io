---
layout: leetcode-entry
title: "744. Find Smallest Letter Greater Than Target"
permalink: "/leetcode/problem/2023-06-09-744-find-smallest-letter-greater-than-target/"
leetcode_ui: true
entry_slug: "2023-06-09-744-find-smallest-letter-greater-than-target"
---

[744. Find Smallest Letter Greater Than Target](https://leetcode.com/problems/find-smallest-letter-greater-than-target/) easy
[blog post](https://leetcode.com/problems/find-smallest-letter-greater-than-target/solutions/3616091/kotlin-binarysearch/)
[substack](https://dmitriisamoilenko.substack.com/p/09062023-744-find-smallest-letter?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/240
#### Problem TLDR
Lowest char greater than `target`.
#### Intuition
In a sorted array, we can use the Binary Search.

#### Approach
For more robust code:
* use inclusive `lo` and `hi`
* check the last condition `lo == hi`
* always move `lo` or `hi`
* always write a good result `res = ...`
* safely compute `mid`
#### Complexity
- Time complexity:
$$O(log(n))$$
- Space complexity:
$$O(1)$$

#### Code

```kotlin

fun nextGreatestLetter(letters: CharArray, target: Char): Char {
    var res = letters[0]
    var lo = 0
    var hi = letters.lastIndex
    while (lo <= hi) {
        val mid = lo + (hi - lo) / 2
        if (letters[mid] > target) {
            hi = mid - 1
            res = letters[mid]
        } else lo = mid + 1
    }
    return res
}

```

