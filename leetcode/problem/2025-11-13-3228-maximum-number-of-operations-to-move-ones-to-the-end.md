---
layout: leetcode-entry
title: "3228. Maximum Number of Operations to Move Ones to the End"
permalink: "/leetcode/problem/2025-11-13-3228-maximum-number-of-operations-to-move-ones-to-the-end/"
leetcode_ui: true
entry_slug: "2025-11-13-3228-maximum-number-of-operations-to-move-ones-to-the-end"
---

[3228. Maximum Number of Operations to Move Ones to the End](https://leetcode.com/problems/maximum-number-of-operations-to-move-ones-to-the-end/description/) medium
[blog post](https://leetcode.com/problems/maximum-number-of-operations-to-move-ones-to-the-end/solutions/7345779/kotlin-rust-by-samoylenkodmitry-o73r/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13112025-3228-maximum-number-of-operations?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Ds7Fgm2Wppo)

![52a8d213-8db3-4706-bf6f-5d9e450d8007 (1).webp](/assets/leetcode_daily_images/37572253.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1172

#### Problem TLDR

Max steps to move all ones to the right #medium

#### Intuition

```j
    // 1010101010
    // 1010101001
    // 1010100011
    // 1010000111
    // 1000001111
    // 0000011111
    //
    //  a b c d e
    // 1010101010
    // 0110101010 a-1
    // 0101101010 b-1
    // 0011101010 b-2
    // 0011011010 c-1
    // 0010111010 c-2
    // 0001111010 c-3
    // 0001110110 d-1
    // 0001101110 d-2
    // 0001011110 d-3
    // 0000111110 d-5
    // 0000111110 e-5 bubble '0' to the left
    // go from left to right, each zero gives +(number of ones)
    // dedup zeros
```
Go from left to right and count ones; each zero adds +count_ones

#### Approach

* skip duplicate consequent zeros

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 39ms
fun maxOperations(s: String) = s.indices.fold(0 to 0) {(o,r),i->
    o+(s[i]-'0') to if (s[i]>'0'||i>0 && s[i]==s[i-1]) r else r+o}.second
```
```rust
// 0ms
    pub fn max_operations(s: String) -> i32 {
        let (mut o, s) = (0, s.as_bytes());
        (0..s.len()).map(|i| { o += (s[i] - b'0') as i32;
            if i>0&&s[i]==s[i-1]||s[i]==b'1'{0}else{o}
        }).sum()
    }
```

