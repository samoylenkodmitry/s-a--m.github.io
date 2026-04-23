---
layout: leetcode-entry
title: "1442. Count Triplets That Can Form Two Arrays of Equal XOR"
permalink: "/leetcode/problem/2024-05-30-1442-count-triplets-that-can-form-two-arrays-of-equal-xor/"
leetcode_ui: true
entry_slug: "2024-05-30-1442-count-triplets-that-can-form-two-arrays-of-equal-xor"
---

[1442. Count Triplets That Can Form Two Arrays of Equal XOR](https://leetcode.com/problems/count-triplets-that-can-form-two-arrays-of-equal-xor/description/) medium
[blog post](https://leetcode.com/problems/count-triplets-that-can-form-two-arrays-of-equal-xor/solutions/5229164/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/30052024-1442-count-triplets-that?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/UfCX9AnqxUM)
![2024-05-30_07-53.webp](/assets/leetcode_daily_images/488cdddd.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/622

#### Problem TLDR

Number `(i,j,k)` where `xor arr[i..j] = xor arr[j..k]` #medium #bit_manipulation

#### Intuition

Start with the brute-force solution, it will be accepted.
```j
                for (j in i + 1..k)
                    a = a ^ arr[j]
                    b = ikXor ^ a
                    if (a == b) res++

```
Some optimizations:
* we have precomputed total xor between `i..k` and now if `a = xor [i..j - 1]` then `b = xor [i..k] ^ a`.

Let's inline `a` and `b` in the `if (a == b)` equation:
```
if (a ^ arr[j] == ikXor ^ (a ^ arr[j])) ...
```
We can safely remove `^ a ^ arr[j]` from the left and the right parts, leaving it like `if (0 == ikXor)`. As this now independent of `j`, we can just collapse the third loop into ` if (0 == ikXor) res += k - i`.

(There is one more optimization possible: store xors prefixes count in a HashMap, this will reduce the time to O(n))

#### Approach

Using `sumOf` and `.map().sum()` helps to reduce some lines of code.

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun countTriplets(arr: IntArray): Int =
        arr.indices.sumOf { i ->
            var ikXor = 0
            (i..<arr.size).sumOf { k ->
                ikXor = ikXor xor arr[k]
                if (0 == ikXor) k - i else 0
            }
        }

```
```rust

    pub fn count_triplets(arr: Vec<i32>) -> i32 {
        (0..arr.len()).map(|i| {
            let mut ik_xor = 0;
            (i..arr.len()).map(|k| {
                ik_xor ^= arr[k];
                if ik_xor == 0 { k - i } else { 0 }
            }).sum::<usize>()
        }).sum::<usize>() as _
    }

```

