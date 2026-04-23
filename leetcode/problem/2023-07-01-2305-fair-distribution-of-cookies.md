---
layout: leetcode-entry
title: "2305. Fair Distribution of Cookies"
permalink: "/leetcode/problem/2023-07-01-2305-fair-distribution-of-cookies/"
leetcode_ui: true
entry_slug: "2023-07-01-2305-fair-distribution-of-cookies"
---

[2305. Fair Distribution of Cookies](https://leetcode.com/problems/fair-distribution-of-cookies/description/) medium
[blog post](https://leetcode.com/problems/fair-distribution-of-cookies/solutions/3702635/kotln-backtrack/)
[substack](https://dmitriisamoilenko.substack.com/p/1072023-2305-fair-distribution-of?sd=pf)
![image.png](/assets/leetcode_daily_images/8e3128ec.webp)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/262

#### Problem TLDR
`Min` of the `max` distributing `n` cookies to `k` children
#### Intuition
Search all possible ways to give current cookie to one of the children. Backtrack sums and calculate the result.

#### Approach
Just DFS

#### Complexity

- Time complexity:
$$O(k^n)$$

- Space complexity:
$$O(2^n)$$

#### Code

```kotlin

fun distributeCookies(cookies: IntArray, k: Int): Int {
    fun dfs(pos: Int, children: IntArray): Int {
        if (pos == cookies.size) return if (children.contains(0)) -1 else children.max()!!
        var min = -1
        for (i in 0 until k) {
            children[i] += cookies[pos]
            val res = dfs(pos + 1, children)
            if (res != -1) min = if (min == -1) res else minOf(min, res)
            children[i] -= cookies[pos]
        }
        return min
    }
    return dfs(0, IntArray(k))
}

```

