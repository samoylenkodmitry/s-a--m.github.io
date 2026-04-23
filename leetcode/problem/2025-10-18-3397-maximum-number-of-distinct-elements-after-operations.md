---
layout: leetcode-entry
title: "3397. Maximum Number of Distinct Elements After Operations"
permalink: "/leetcode/problem/2025-10-18-3397-maximum-number-of-distinct-elements-after-operations/"
leetcode_ui: true
entry_slug: "2025-10-18-3397-maximum-number-of-distinct-elements-after-operations"
---

[3397. Maximum Number of Distinct Elements After Operations](https://leetcode.com/problems/maximum-number-of-distinct-elements-after-operations/description) medium
[blog post](https://leetcode.com/problems/maximum-number-of-distinct-elements-after-operations/solutions/7283635/kotlin-rust-by-samoylenkodmitry-zgkh/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18102025-3397-maximum-number-of-distinct?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/kZ6aPX5ICKo)

![2d4101ab-32f3-466d-ab2f-475bd5d27b01 (1).webp](/assets/leetcode_daily_images/29a2646f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1146

#### Problem TLDR

Distincts count after adding -k..k to each #medium

#### Intuition

Sort.
There is a window we can take from duplicates: -k, -k+1, ..., k-1, k.
Slide from the left, greedily apply lowest possible change to the number. Update max of used values.

#### Approach

* don't stop when current value is out of range, the next can be bigger

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 515ms
    fun maxDistinctElements(n: IntArray, k: Int): Int {
        var m = Int.MIN_VALUE; n.sort()
        return n.count { val r = m < it + k; if (m < it+k) m = max(m+1,it-k); r }
    }

```
```rust

// 20ms
    pub fn max_distinct_elements(n: Vec<i32>, k: i32) -> i32 {
        let mut m = i32::MIN;
        n.iter().sorted().filter(|&x| { let r = m < x+k; if (r) { m = (m+1).max(x-k)}; r }).count() as _
    }

```

