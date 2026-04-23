---
layout: leetcode-entry
title: "2598. Smallest Missing Non-negative Integer After Operations"
permalink: "/leetcode/problem/2025-10-16-2598-smallest-missing-non-negative-integer-after-operations/"
leetcode_ui: true
entry_slug: "2025-10-16-2598-smallest-missing-non-negative-integer-after-operations"
---

[2598. Smallest Missing Non-negative Integer After Operations](https://leetcode.com/problems/smallest-missing-non-negative-integer-after-operations/description/) medium
[blog post](https://leetcode.com/problems/smallest-missing-non-negative-integer-after-operations/solutions/7279252/kotlin-rust-by-samoylenkodmitry-2bux/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16102025-2598-smallest-missing-non?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/NxasuZ06jZ8)

![cb628084-0ca6-4203-becb-a78cbcd2ab7b (1).webp](/assets/leetcode_daily_images/dd720f30.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1144

#### Problem TLDR

First positive number can't be build by adding/subtracting value #medium #hashmap

#### Intuition

Iterate from zero and look for the reminder. Use it or stop.

#### Approach

* corner case: negatives
* we can use array size of `v` as a frequency map

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 135ms
    fun findSmallestInteger(n: IntArray, v: Int): Int {
        val m = n.groupBy{ (it%v+v)%v }.mapValues{ it.value.size }.toMutableMap()
        return (0..n.size).first { val c = m[it%v]?:0; m[it%v]=c-1; c < 1 }
    }

```
```rust

// 0ms
    pub fn find_smallest_integer(n: Vec<i32>, v: i32) -> i32 {
        let mut f = vec![0; v as usize]; for &x in &n { f[((x%v+v)%v) as usize] += 1 }
        (0..=n.len()).find(|x| { let c = f[x%v as usize]; f[x%v as usize] -= 1; c < 1 }).unwrap() as _
    }

```

