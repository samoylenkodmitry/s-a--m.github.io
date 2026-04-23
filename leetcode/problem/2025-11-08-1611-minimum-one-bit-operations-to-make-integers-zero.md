---
layout: leetcode-entry
title: "1611. Minimum One Bit Operations to Make Integers Zero"
permalink: "/leetcode/problem/2025-11-08-1611-minimum-one-bit-operations-to-make-integers-zero/"
leetcode_ui: true
entry_slug: "2025-11-08-1611-minimum-one-bit-operations-to-make-integers-zero"
---

[1611. Minimum One Bit Operations to Make Integers Zero](https://leetcode.com/problems/minimum-one-bit-operations-to-make-integers-zero/description) hard
[blog post](https://leetcode.com/problems/minimum-one-bit-operations-to-make-integers-zero/solutions/7334462/kotlin-rust-by-samoylenkodmitry-pnon/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/08112025-1611-minimum-one-bit-operations?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/8fU_ZJzmS6Q)

![649c14d6-6dcd-4881-ab28-3aa8be3f5f31 (1).webp](/assets/leetcode_daily_images/38dad101.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1167

#### Problem TLDR

Min steps to make zero by xor 1 or xor rightmost+1 #hard #bits

#### Intuition

Didn't solve myself.
```j
   // 000010101
    //        **
    // 000100000 1
    // 000100001 2
    // 000100011 3
    // 000100010 4
    // 000100110 5
    // 000100101 6
    // 000100100 7
    // 000101100 8
    // 000101101 9
    // 000101111 10
    // 000101110 11
    // 000101010 12
    // 000101011 13
    // 000101001 14
    // 000101000 15
    // 000111000 16
    // 000111001 17
    // 000111011 18
    // 000111010 19
    // 000111110 20
    // 000111111 21
    // 000111101 22
    // 000111100 23
    // 000110100 24
    // 000110101 25
    // 000110111 26
    // 000110110 27
    // 000110010 28
    // 000110011 29
    // 000110001 30
    // 000110000 31
    // 000010000 0
    // 000010001 1
    // 000010011 2
    // 000010010 3
    // 000010110 4
    // 000010111 5
    // 000010101 6
    // 000010100 7
    // 000011100 8
    // 000011101 9
    // 000011111 10
    // 000011110 11
    // 000011010 12
    // 000011011 13
    // 000011001 14
    // 000011000 15
    // 000001000 0
    // 000001001 1
    // 000001011 2
    // 000001010 3
    // 000001110 4
    // 000001111 5
    // 000001101 6
    // 000001100 7
    // 000000100 0
    // 000000101 1
    // 000000111 2
    // 000000110 3
    // 000000010 0
    // 000000011 1
    // 000000001 0
    //
    // the brute force gives TLE (19 minute)

    // bit 8
    // bit 7   255      128
    // bit 6 - 127 step  64
    // bit 5 - 63 step  32
    // bit 4 - 31 steps 16
    // bit 3 - 15 steps  8
    // bit 2 - 7 steps  4
    // bit 1 - 3 steps  2
    // bit 0 - 1 steps  1
    // the law bit - 2^(bit+1)-1 it is to make zero, not just remove bit

    // but what if we start not from a single bit? any additional bit decrease the steps needed
    // or, its all steps for the next bit plus tail

    // maybe brute force until find some power of two? - TLE
    // Hint 2 (45 minute) useless, i don't know how to transition from a single bit to several bits
    // ok, hint from discussion: the tail bits treated like separate subtractions
    //
    // 1011 = 1000 + 0010 + 0001
    //        3        1       0
    //        a        b       c
    //        a    - ( b   -   c)

```

1. The law for the single bit set is `2^(bit+1)-1` - just check testcases with 2,4,8,16,32,64,..
2. Solve for the most significant bit. The steps *already* include the tail, so subtract the tail. Do the recursion.

#### Approach

* to find the *2* intuition you can spend a day or month of thinking

#### Complexity

- Time complexity:
$$O(log(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 0ms
    fun minimumOneBitOperations(n: Int): Int {
        if (n == 0) return 0
        val bit = n.takeHighestOneBit()
        return (bit shl 1) - 1 - minimumOneBitOperations(n xor bit)
    }
```
```rust
// 0ms
    pub fn minimum_one_bit_operations(n: i32) -> i32 {
        if n < 1 { return 0 }
        let b = 1 << (31 - n.leading_zeros());
        (b << 1) - 1 - Self::minimum_one_bit_operations(n^b)
    }
```

