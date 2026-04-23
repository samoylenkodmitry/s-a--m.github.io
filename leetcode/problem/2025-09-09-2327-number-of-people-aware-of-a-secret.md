---
layout: leetcode-entry
title: "2327. Number of People Aware of a Secret"
permalink: "/leetcode/problem/2025-09-09-2327-number-of-people-aware-of-a-secret/"
leetcode_ui: true
entry_slug: "2025-09-09-2327-number-of-people-aware-of-a-secret"
---

[2327. Number of People Aware of a Secret](https://leetcode.com/problems/number-of-people-aware-of-a-secret/description) medium
[blog post](https://leetcode.com/problems/number-of-people-aware-of-a-secret/solutions/7171712/kotlin-by-samoylenkodmitry-nbg6/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/09092025-2327-number-of-people-aware?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/35yCdQG4Tg0)

![1.webp](/assets/leetcode_daily_images/d7523177.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1107

#### Problem TLDR

People knowing a secret at day n, keeping delay then spread and forget #medium #simulation

#### Intuition

```j
    // time, delay=1, forget = 3
    // 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
    // a a a a
    //   b b b b
    //     c c c c
    //       d d d d
    //       e
    //       f
    //       \
    //        one from a(d), one from b(e), one from c(f)
    //        3 active, so +3 passive (+delay day +3 active)

```

Maintain diff array for
* changing active spreaders
* changing knowers

Another intuition from lee: dp[i] is how many new people at that day. From that angle we know dp[i-forget] is how many people will forget today, dp[i-(forget-delay)] is how many people became active today. Very useful.

#### Approach

* the main difficulty is to find a good angle to look at this problem

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 4ms
    fun peopleAwareOfSecret(n: Int, delay: Int, forget: Int): Int {
        val M = 1000000007L
        val know = LongArray(n+forget+1); val active = LongArray(n+forget+1)
        know[delay] = 1; know[1+forget] = -1; active[delay] = 1; active[1+forget] = -1
        for (d in 1+delay..n) {
            active[d] = (active[d-1] + active[d] + M) % M
            active[d+delay] += active[d]; active[d+forget] -= active[d]
            know[d] = (know[d-1] + know[d] + active[d] + M) % M
            know[d+forget] -= active[d]
        }
        return know[n].toInt()
    }

```

