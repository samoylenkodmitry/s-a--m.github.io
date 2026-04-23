---
layout: leetcode-entry
title: "976. Largest Perimeter Triangle"
permalink: "/leetcode/problem/2025-09-28-976-largest-perimeter-triangle/"
leetcode_ui: true
entry_slug: "2025-09-28-976-largest-perimeter-triangle"
---

[976. Largest Perimeter Triangle](https://leetcode.com/problems/largest-perimeter-triangle/description/) easy
[blog post](https://leetcode.com/problems/largest-perimeter-triangle/solutions/7230895/kotlin-rust-by-samoylenkodmitry-ml7d/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28092025-976-largest-perimeter-triangle?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/JLBa2i5asi0)

![1.webp](/assets/leetcode_daily_images/df9d602b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1126

#### Problem TLDR

Max triangle perimeter from array of lengths #easy #sliding_window

#### Intuition

Sort. Consider every 3-sliding window from biggest: when `a+b <= c` move next, discard the `c`, otherwise finish.
When `c` is bigger than closes `a+b`, then it will be bigger than any other pair sum.

#### Approach

* this is not an easy problem

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$ or O(1)

#### Code

```kotlin

// 81ms
    fun largestPerimeter(n: IntArray) =
        max(0, n.sorted().windowed(3)
        .maxOf{(a,b,c) -> (a+b+c)*(a+b).compareTo(c)})

```

```rust

// 0ms
    pub fn largest_perimeter(n: Vec<i32>) -> i32 {
        n.iter().sorted_by_key(|&x|-x).tuple_windows()
        .find(|&(c,b,a)| a+b>*c).map_or(0, |(a,b,c)| a+b+c)
    }

```

