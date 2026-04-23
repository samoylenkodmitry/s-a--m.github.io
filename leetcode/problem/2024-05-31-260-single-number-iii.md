---
layout: leetcode-entry
title: "260. Single Number III"
permalink: "/leetcode/problem/2024-05-31-260-single-number-iii/"
leetcode_ui: true
entry_slug: "2024-05-31-260-single-number-iii"
---

[260. Single Number III](https://leetcode.com/problems/single-number-iii/description/) medium
[blog post](https://leetcode.com/problems/single-number-iii/solutions/5233996/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/31052024-260-single-number-iii?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/A5rVOkz0If8)
![2024-05-31_08-32.webp](/assets/leetcode_daily_images/abf852c1.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/623

#### Problem TLDR

Two not duplicated numbers from array #medium #bit_manipulation

#### Intuition

The first idea is to `xor` the array, `xor[..] = a ^ b`.
However from that point there is no clear path to what can be done next.

(I personally gave up and go to the discussion section)

The hint: each `1` bit in the xor result of `a ^ b` means that in that bit `a` is different than `b`. We can split all the numbers in array by this bit: one group will contain `a` and some duplicates, another group will contain `b` and some other remaining duplicates. Those duplicates can be xored and `a` and `b` distilled.

```j
    // a b cc dd   xor[..] = a ^ b
    // 1 2 1 3 2 5
    // 1  01
    // 2  10
    // 1  01
    // 3  11
    // 2  10
    // 5 101
    //
    // x 110
    //     *   (same bits in a and b)
    //    *    1 1 5       vs   2 3 2
    //   *     1 2 1 3 2   vs   5
```

#### Approach

Some tricks:
* `first` and `find` operators in Kotlin and Rust
* conversion of `boolean` to `usize` in Rust

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun singleNumber(nums: IntArray): IntArray {
        var x = 0; for (n in nums) x = x xor n
        return (0..31).first { x and (1 shl it) != 0 }.let {
            var a = 0; var b = 0
            for (n in nums) if ((n and (1 shl it)) != 0)
                a = a xor n else b = b xor n
            intArrayOf(a, b)
        }
    }

```
```rust

    pub fn single_number(nums: Vec<i32>) -> Vec<i32> {
        let (mut x, mut r) = (0, vec![0, 0]); for &n in &nums { x ^= n }
        let bit = (0..32).find(|&bit| x & (1 << bit) != 0).unwrap();
        for &n in &nums { r[(n & (1 << bit) != 0) as usize] ^= n }; r
    }

```

