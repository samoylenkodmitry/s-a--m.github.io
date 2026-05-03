---
layout: leetcode-entry
title: "788. Rotated Digits"
permalink: "/leetcode/problem/2026-05-02-788-rotated-digits/"
leetcode_ui: true
entry_slug: "2026-05-02-788-rotated-digits"
---

[788. Rotated Digits](https://leetcode.com/problems/rotated-digits/solutions/8128761/kotlin-rust-by-samoylenkodmitry-sthe/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/02052026-788-rotated-digits?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/HvF1tGILl4w)

https://dmitrysamoylenko.com/leetcode/

![02.05.2026.webp](/assets/leetcode_daily_images/02.05.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1346

#### Problem TLDR

Count numbers in 0..n with mirrored digits 2-5 6-9, 0,1,8, but not 3,4,7

#### Intuition

Brute-force is accepted.
The logN solution:
* each digit is a start of the subtree
* in the subtree we have seven numbers total 0,1,2,5,6,8,9 and three numbers we should avoid to form tail from them - 0,1,8
* the tails length is K, means total numbers count is 7^K, and to avoid is 3^K
* if prefix has good number 2-5,6-9 then suffix can take all 7^K in its tail

#### Approach

* regex [0125689]* means any good prefix, [2569] must have any of this numbers, [0125689]* any good suffix

#### Complexity

- Time complexity:
$$O(n|logN)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun rotatedDigits(n: Int) = (1..n).count {
        Regex("^[0125689]*[2569][0125689]*$") in "$it"
    }
```
```rust
    pub fn rotated_digits(n: i32) -> i32 {
        let s = n.to_string(); let l = s.len() as u32;
        let (mut p7, mut p3, mut r, mut m) = (7_i32.pow(l), 3_i32.pow(l), 0, 0);
        for c in s.bytes() {
            p7 /= 7; p3 /= 3; let c = (c - 48) as i32;
            for d in 0..c
                { r += (1-(152>>d&1))*(p7 - if (m | 1 << d) & 612 > 0 { 0 } else { p3 }) }
            m |= 1 << c; if m & 152 > 0 { return r; }
        } r + (m & 612 > 0) as i32
    }
```

