---
layout: leetcode-entry
title: "With All Set Bits"
permalink: "/leetcode/problem/2025-10-29-with-all-set-bits/"
leetcode_ui: true
entry_slug: "2025-10-29-with-all-set-bits"
---

[With All Set Bits](https://leetcode.com/problems/smallest-number-with-all-set-bits/description/) easy
[blog post](https://leetcode.com/problems/smallest-number-with-all-set-bits/solutions/7310067/kotlin-rust-by-samoylenkodmitry-wd82/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29102025-with-all-set-bits?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/S3WRk2iWJXM)

![56938ebc-e4f5-4d0c-b54a-5848bbd610b3 (1).webp](/assets/leetcode_daily_images/d5afaf13.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1157

#### Problem TLDR

Next  all bits set number #easy #bits

#### Intuition

HighestOneBit shl 1 - 1
```j
101
100 highest one bit
1000 shl 1
0111  -1
```

#### Approach

* how to find highestOneBit?
* we can brute force too `(n..2*n).first { (it+1) and it == 0 }`

#### Complexity

- Time complexity:
$$O(1)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 0ms
    fun smallestNumber(n: Int) =
        (n.takeHighestOneBit() shl 1) - 1

```
```rust
// 0ms
    pub fn smallest_number(n: i32) -> i32 {
        i32::MAX >> n.leading_zeros() - 1
    }

```

