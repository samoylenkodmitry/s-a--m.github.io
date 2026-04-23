---
layout: leetcode-entry
title: "455. Assign Cookies"
permalink: "/leetcode/problem/2024-01-01-455-assign-cookies/"
leetcode_ui: true
entry_slug: "2024-01-01-455-assign-cookies"
---

[455. Assign Cookies](https://leetcode.com/problems/assign-cookies/description/) easy
[blog post](https://leetcode.com/problems/assign-cookies/solutions/4486297/kotlin-sort/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/1012024-455-assign-cookies?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/Y5ARRSdTOEY)
![image.png](/assets/leetcode_daily_images/a18e8779.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/458

#### Problem TLDR

Max count of greedy children g[Int] to assign cookies with sizes s[Int].

#### Intuition

The optimal way to assign cookies is to start with less greed. We can put cookies and children in two PriorityQueues or just sort two arrays and maintain two pointers.

#### Approach

* PriorityQueue is a more error-safe solution, also didn't modify the input.
* Careful with the pointers, check yourself with simple examples: `g=[1] s=[1]`, `g=[2] s=[1]`

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun findContentChildren(g: IntArray, s: IntArray): Int {
    g.sort()
    s.sort()
    var j = 0
    return g.count {
      while (j < s.size && s[j] < it ) j++
      j++ < s.size
    }
  }

```

