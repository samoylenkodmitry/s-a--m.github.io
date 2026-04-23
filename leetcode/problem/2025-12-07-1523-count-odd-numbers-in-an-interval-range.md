---
layout: leetcode-entry
title: "1523. Count Odd Numbers in an Interval Range"
permalink: "/leetcode/problem/2025-12-07-1523-count-odd-numbers-in-an-interval-range/"
leetcode_ui: true
entry_slug: "2025-12-07-1523-count-odd-numbers-in-an-interval-range"
---

[1523. Count Odd Numbers in an Interval Range](https://leetcode.com/problems/count-odd-numbers-in-an-interval-range/description/) easy
[blog post](https://leetcode.com/problems/count-odd-numbers-in-an-interval-range/solutions/7397704/kotlin-rust-by-samoylenkodmitry-gw8p/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/07122025-1523-count-odd-numbers-in?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/qjWcxt3lY18)

![ff0763dc-acf5-46ba-a96b-cd897810f634 (1).webp](/assets/leetcode_daily_images/338b4751.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1196

#### Problem TLDR

Odds l..h #easy

#### Intuition

Look at total numbers count `d`:
* even d - just divide by 2
* odd d - divide by 2 plus look at first number if it is odd

```j
    // 1 2      2-1+1=2,  l=o, h = e, 2/2
    // 1 2 3 4  4, l=o h=e 4/2
    // 1 2 3    3   l=o h=o   1+3/2
    // 2 3      2   l=e    2/2
    // 2 3 4 5  4   l=e    4/2
    // 3 4 5    3   l=o    1+3/2
    // 2 3 4    3   l=e    3/2
    // 3 4 5 6  4   l=o    4/2
```

#### Approach

* or brillian lee compression of this logic: (h+1)/2-l/2

#### Complexity

- Time complexity:
$$O(1)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 76ms
    fun countOdds(l: Int, h: Int) =
        (h-l+1)/2 + (l%2)*((h-l+1)%2)
```
```rust
// 0ms
    pub fn count_odds(l: i32, h: i32) -> i32 {
       (h+1)/2 - l/2
    }
```

