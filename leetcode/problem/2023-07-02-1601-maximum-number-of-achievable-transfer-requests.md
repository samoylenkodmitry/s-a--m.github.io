---
layout: leetcode-entry
title: "1601. Maximum Number of Achievable Transfer Requests"
permalink: "/leetcode/problem/2023-07-02-1601-maximum-number-of-achievable-transfer-requests/"
leetcode_ui: true
entry_slug: "2023-07-02-1601-maximum-number-of-achievable-transfer-requests"
---

[1601. Maximum Number of Achievable Transfer Requests](https://leetcode.com/problems/maximum-number-of-achievable-transfer-requests/description/) hard
[blog post](https://leetcode.com/problems/maximum-number-of-achievable-transfer-requests/solutions/3706324/kotlin-bitmask/)
[substack](https://dmitriisamoilenko.substack.com/p/2072023-1601-maximum-number-of-achievable?sd=pf)
![image.png](/assets/leetcode_daily_images/d865b09a.webp)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/263
#### Problem TLDR
Max edges to make all counts `in == out` edges in graph
#### Intuition
Let's observe some examples:
![image.png](/assets/leetcode_daily_images/f438b1f4.webp)

All requests are valid if count of incoming edges are equal to outcoming.
One possible solution is to just check each combination of edges.
#### Approach
Let's use bitmask to traverse all combinations, as total number `16` can fit in `Int`

#### Complexity

- Time complexity:
$$O(n2^r)$$

- Space complexity:
$$O(n2^r)$$

#### Code

```kotlin

fun maximumRequests(n: Int, requests: Array<IntArray>): Int =
    (0..((1 shl requests.size) - 1)).filter { mask ->
        val fromTo = IntArray(n)
        requests.indices.filter { ((1 shl it) and mask) != 0 }.forEach {
            val (from, to) = requests[it]
            fromTo[from] -= 1
            fromTo[to] += 1
        }
        fromTo.all { it == 0 }
    }.map { Integer.bitCount(it) }.max()!!

```

