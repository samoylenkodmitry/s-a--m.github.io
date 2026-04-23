---
layout: leetcode-entry
title: "28. Find the Index of the First Occurrence in a String"
permalink: "/leetcode/problem/2023-03-03-28-find-the-index-of-the-first-occurrence-in-a-string/"
leetcode_ui: true
entry_slug: "2023-03-03-28-find-the-index-of-the-first-occurrence-in-a-string"
---

[28. Find the Index of the First Occurrence in a String](https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/description/) medium

[blog post](https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/solutions/3250975/kotlin-rolling-hash/)

```kotlin

fun strStr(haystack: String, needle: String): Int {
    // f(x) = a + 32 * f(x - 1)
    // abc
    // f(a) = a + 0
    // f(ab) = b + 32 * (a + 0)
    // f(abc) = c + 32 * (b + 32 * (a + 0))
    //
    // f(b) = b + 0
    // f(bc) = c + 32 * (b + 0)
    //
    // f(abc) - f(bc) = 32^0*c + 32^1*b + 32^2*a - 32^0*c - 32^1*b = 32^2*a
    // f(bc) = f(abc) - 32^2*a
    var needleHash = 0L
    needle.forEach { needleHash = it.toLong() + 32L * needleHash }
    var currHash = 0L
    var pow = 1L
    repeat(needle.length) { pow *= 32L}
    for (curr in 0..haystack.lastIndex) {
        currHash = haystack[curr].toLong() + 32L * currHash
        if (curr >= needle.length)
        currHash -= pow * haystack[curr - needle.length].toLong()
        if (curr >= needle.lastIndex
        && currHash == needleHash
        && haystack.substring(curr - needle.lastIndex, curr + 1) == needle)
        return curr - needle.lastIndex
    }
    return -1
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/136
#### Intuition
There is a `rolling hash` technique: you can compute hash for a sliding window using O(1) additional time.
Consider the math behind it:

```

// f(x) = a + 32 * f(x - 1)
// abc
// f(a) = a + 0
// f(ab) = b + 32 * (a + 0)
// f(abc) = c + 32 * (b + 32 * (a + 0))
//
// f(b) = b + 0
// f(bc) = c + 32 * (b + 0)
//
// f(abc) - f(bc) = 32^0*c + 32^1*b + 32^2*a - 32^0*c - 32^1*b = 32^2*a
// f(bc) = f(abc) - 32^2*a

```

Basically, you can subtract `char * 32^window_length` from the lower side of the sliding window.

#### Approach
* carefull with indexes
#### Complexity
- Time complexity:
$$O(n)$$, if our hash function is good, we good
- Space complexity:
$$O(n)$$, for substring, can be improved to O(1)

