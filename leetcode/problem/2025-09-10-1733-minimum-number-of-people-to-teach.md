---
layout: leetcode-entry
title: "1733. Minimum Number of People to Teach"
permalink: "/leetcode/problem/2025-09-10-1733-minimum-number-of-people-to-teach/"
leetcode_ui: true
entry_slug: "2025-09-10-1733-minimum-number-of-people-to-teach"
---

[1733. Minimum Number of People to Teach](https://leetcode.com/problems/minimum-number-of-people-to-teach/description/) medium
[blog post](https://leetcode.com/problems/minimum-number-of-people-to-teach/solutions/7174800/kotlin-by-samoylenkodmitry-pqa7/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10092025-1733-minimum-number-of-people?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/BWxJDjXCuZ0)

![1.webp](/assets/leetcode_daily_images/fcad35dd.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1108

#### Problem TLDR

Min friends to teach common language to communicate in graph #medium

#### Intuition

Only one language is allowed.

```j
    // user1: knows lang1, friends with user2,user3
    // user2: knows lang2, friends with user1,user3
    // user3: knows lang1,lang2, friends with user2

    // users graph:
    //     l1    l2     l1,l2
    //      u1---u2---u3
    //       \________/
    //
    // u1 & u2 can't communicate
    // u1 & u3 can
    // u2 & u3 can
    //
    // so u1 should learn any langs of u2
    // or
    // u2 should learn any langs of u1
    //
    // and we should make minimum users to teach

    //   [2] [3]  [1,2]
    //     1--4--3
    //      \   /         it is 3 components: 1, 4, 2-3
    //        2
    //         [1,3]
    //
    //    1 [+3] and 4 [+2]
    //
    // i don't get the optimal algo, look at hint (21 minute)
    //
    // so are users should talk each-to-each, event not direct friends?
    // why graph then?
```

I used the hint: just brute force all languages.

#### Approach

* the main difficulty is to understand the problem
* only consider non-communicating pairs
* find most common language in non-communicated people

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 83ms
    fun minimumTeachings(n: Int, ls: Array<IntArray>, fs: Array<IntArray>): Int {
        val ls = ls.map { it.toSet() }; val fs = fs.map { it.toList() }
        var f = fs.filter { (a, b) -> !ls[b-1].any { it in ls[a-1] }}.flatten().toSet()
        return f.size - (f.flatMap { ls[it-1] }.groupBy{it}.maxOfOrNull{it.value.size}?:0)
    }

```

