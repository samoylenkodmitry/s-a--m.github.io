---
layout: leetcode-entry
title: "3315. Construct the Minimum Bitwise Array II"
permalink: "/leetcode/problem/2026-01-21-3315-construct-the-minimum-bitwise-array-ii/"
leetcode_ui: true
entry_slug: "2026-01-21-3315-construct-the-minimum-bitwise-array-ii"
---

[3315. Construct the Minimum Bitwise Array II](https://leetcode.com/problems/construct-the-minimum-bitwise-array-ii/description/) medium
[blog post](https://leetcode.com/problems/construct-the-minimum-bitwise-array-ii/solutions/7512180/kotlin-rust-by-samoylenkodmitry-e9c1/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21012026-3315-construct-the-minimum?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/0Idxr_-Jqdc)

![72fdf644-52d3-4fde-86e4-2bdddf19affe (1).webp](/assets/leetcode_daily_images/81f32d9b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1244

#### Problem TLDR

Reverse operation x or (x+1) #medium #bits

#### Intuition

Forward operation: flips first one-bit in tail of ones.
Reverse: flip rightmost zero bit.

```j
    /*
    3 : 11 -> 1 : 1
5 : 101 -> 4 : 100
7 : 111 -> 3 : 11
11 : 1011 -> 9 : 1001
13 : 1101 -> 12 : 1100
17 : 10001 -> 16 : 10000
19 : 10011 -> 17 : 10001
23 : 10111 -> 19 : 10011
29 : 11101 -> 28 : 11100
31 : 11111 -> 15 : 1111
37 : 100101 -> 36 : 100100
41 : 101001 -> 40 : 101000
43 : 101011 -> 41 : 101001
47 : 101111 -> 39 : 100111
53 : 110101 -> 52 : 110100
59 : 111011 -> 57 : 111001
61 : 111101 -> 60 : 111100
67 : 1000011 -> 65 : 1000001
71 : 1000111 -> 67 : 1000011
73 : 1001001 -> 72 : 1001000
79 : 1001111 -> 71 : 1000111
83 : 1010011 -> 81 : 1010001
89 : 1011001 -> 88 : 1011000
97 : 1100001 -> 96 : 1100000
*/
// 79 : 1001111 -> 71 : 1000111
// +1   1010000
//  &   1000000
//  +  10001111
// /2   1000111

//
// 97 : 1100001 -> 96 : 1100000
// +1   1100010
//  &   1100000
//  +                  x+x in binary is 2*x which is shift left by 1
//     11000001        but we preserve a tail as is
// /2   1100000
```

One way: inv, &, /2, xor
Second way: +1, &, +, /2

#### Approach

* or you can just manually find the rightmost zero bit and flip it

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 7ms
    fun minBitwiseArray(n: List<Int>) = n.map {
        if (it==2)-1 else it xor it.inv().takeLowestOneBit()/2
    }
```
```rust
// 0ms
    pub fn min_bitwise_array(n: Vec<i32>) -> Vec<i32> {
        n.iter().map(|&n|if n==2{-1}else{(n+(n&(n+1)))/2}).collect()
    }
```

