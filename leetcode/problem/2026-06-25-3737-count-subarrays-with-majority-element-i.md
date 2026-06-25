---
layout: leetcode-entry
title: "3737. Count Subarrays With Majority Element I"
permalink: "/leetcode/problem/2026-06-25-3737-count-subarrays-with-majority-element-i/"
leetcode_ui: true
entry_slug: "2026-06-25-3737-count-subarrays-with-majority-element-i"
---

[3737. Count Subarrays With Majority Element I](https://leetcode.com/problems/count-subarrays-with-majority-element-i/solutions/8357059/kotlin-rust-by-samoylenkodmitry-fw1g/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/25062026-3737-count-subarrays-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Aua17hAHSsg)

https://dmitrysamoylenko.com/leetcode/

![25.06.2026.webp](/assets/leetcode_daily_images/25.06.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1401

#### Problem TLDR

Subarrays with majority element

#### Intuition

The problem space is small 1000, O(n^2) is accepted

#### Approach

* in the inner loop calculate a running sum

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun countMajoritySubarrays(n: IntArray, t: Int) = n
    .indices.sumOf { i -> var c = 0
        (i downTo 0).count { j ->  if (n[j]==t) c++; c > (i-j+1)/2 }
    }
```
```rust
    pub fn count_majority_subarrays(n: Vec<i32>, t: i32) -> i32 {
        (0..n.len()).map(|i|{ let mut c = 0;
            (0..=i).rev().filter(|&j| {
                if n[j] == t { c += 1 }; c*2 > i-j+1
            }).count() as i32
        }).sum()
    }
```

