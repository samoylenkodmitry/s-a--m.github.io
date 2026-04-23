---
layout: leetcode-entry
title: "762. Prime Number of Set Bits in Binary Representation"
permalink: "/leetcode/problem/2026-02-21-762-prime-number-of-set-bits-in-binary-representation/"
leetcode_ui: true
entry_slug: "2026-02-21-762-prime-number-of-set-bits-in-binary-representation"
---

[762. Prime Number of Set Bits in Binary Representation](https://open.substack.com/pub/dmitriisamoilenko/p/21022026-762-prime-number-of-set?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) easy
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/21022026-762-prime-number-of-set?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21022026-762-prime-number-of-set?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/X63Rkrgew6A)

![5f0418bb-1a61-41b3-ad9d-84bd31aafa6f (1).webp](/assets/leetcode_daily_images/e5d75ead.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1276

#### Problem TLDR

Number in l..r with prime bits count #easy #combinatorics

#### Intuition

Just brute force the range L..R.

#### Approach

* only 2..19 primes, all can fit in bitmask 0xa28ac
* we can solve the problem for each prime: "how many numbers up to R with the same bits count?"
* custom solutin with combinatorics: check every set bit, how many places it can be put in a tail
* Gosper's hack: find next smallest number with the same set bits count

#### Complexity

- Time complexity:
$$O(n)$$ to O(log(n)) for combinatorics

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 1ms
    fun countPrimeSetBits(l: Int, r: Int) =
    intArrayOf(2,3,5,7,11,13,17,19).sumOf { f(r,it)-f(l-1,it) }
    companion object {
        val nCr = Array(22) { IntArray(22) }
        init { for (i in 0..20) {
            nCr[i][0] = 1
            for (j in 1..i) nCr[i][j]=nCr[i-1][j-1]+nCr[i-1][j]
        }}
        fun f(n: Int, k: Int): Int {
            var cnt = 0; var b = 0
            for (i in 20 downTo 0) if (n shr i and 1 > 0)
                if (k-b >= 0) cnt += nCr[i][k-b++]
            if (b == k) cnt++
            return cnt
        }
    }
```
```kotlin
// 86ms
    fun countPrimeSetBits(l: Int, r: Int): Int {
        var cnt = 0
        for (p in listOf(2, 3, 5, 7, 11, 13, 17, 19)) {
            var x = (1 shl p) - 1
            while (x <= r) {
                if (x >= l) ++cnt
                val c = x and -x
                val nextR = x + c
                x = (nextR xor x shr 2) / c or nextR
            }
        }
        return cnt
    }
```
```rust
// 0ms
    pub fn count_prime_set_bits(l: i32, r: i32) -> i32 {
       (l..=r).map(|x| 0xa28ac >> x.count_ones() & 1).sum::<i32>()
    }
```

