---
layout: leetcode-entry
title: "3539. Find Sum of Array Product of Magical Sequences"
permalink: "/leetcode/problem/2025-10-12-3539-find-sum-of-array-product-of-magical-sequences/"
leetcode_ui: true
entry_slug: "2025-10-12-3539-find-sum-of-array-product-of-magical-sequences"
---

[3539. Find Sum of Array Product of Magical Sequences](https://leetcode.com/problems/find-sum-of-array-product-of-magical-sequences/description/) medium
[blog post](https://leetcode.com/problems/find-sum-of-array-product-of-magical-sequences/solutions/7269299/kotlin-rust-by-samoylenkodmitry-as5v/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12102025-3539-find-sum-of-array-product?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/U2ZaPwTYtJM)

![4aab11ea-339f-4961-8990-f511ccadbe08 (1).webp](/assets/leetcode_daily_images/9a343764.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1140

#### Problem TLDR

Product of all `good` subsequences and permutations #hard #combinatorics

#### Intuition

```j
    // by looking at the constraints: full search may be possible
    // take 30 indices out of 50 nums
    // how to use sum(2^seq[i]) = k set bits
    // set bits are the sum of 2^bit1 + 2^bit2 + 2^bit3, where 1,2,3 are the set bits
    // 1..k..m..30
    //
    // after sequence is found, we should count permutations and multiply by product

    // 0 1 2 3 4 5   k=2 m=2
    //               2^0 + 2^1 = 1+2=3 = b011
    //               2^0 + 2^2 = 1+4=5 = b101
    // so it is just any-to-any permutations
    // 0:   1 2 3 4 5
    // 1: 0   2 3 4 5
    // 2: 0 1   3 4 5
    // 3: 0 1 2   4 5
    // 4: 0 1 2 3   5
    // 5: 0 1 2 3 4      5^5=25
    // choose k from n

    // no examples where k < m
    // 0 1 2 3 4 5   k=2 m=3
    // let's write brute-force to see the picture 2^50 too much

    // 39 minute: brute-force works, TLE for n=50,m=30,k=20
    // this can be about combinatorics, let's look for hint
    // 1 dp?, the nums can't be split, so we have to look at m and k
    // m=1,k=1: single bit, single index, powers of two 0 2 4 8 16 32
    // m=2,k=1: single bit, two indices, choose two from previous
    // m=2,k=2: two bits, two indices:
    // 53 minute: ok, i looked at hints and decided to give up

    // basically inline checkmagic and prod compute in the dfs or not

```

#### Approach

* maybe I should learn combinatorics

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 405ms
    fun magicalSum(m: Int, k: Int, nums: IntArray): Int {
        val dp = HashMap<Long, Long>(); val M = 1000000007L
        fun fact(n: Long): Long = if (n == 0L) 1L else (fact(n-1L)*n)%M
        fun pow(a:Long,x:Long):Long=if (x==1L)a else if(x==0L)1L else (pow(a*a%M,x/2)*pow(a,x and 1L))%M
        val fc = LongArray(m + 1); fc[m] = pow(fact(1L* m), M-2); for (i in m-1 downTo 0) fc[i] = (fc[i+1] * (i+1))%M
        fun nCr(n: Int, r: Int):Long = (((fact(1L*n) * fc[r]) % M) * fc[n-r])%M
        fun f(mask: Long, m: Int, k: Int, i: Int): Long = dp.getOrPut(mask*1000000+m*10000+k*100+i) {
            if (m == 0) return@getOrPut if (mask.countOneBits() == k) 1L else 0L
            if (i == nums.size) return@getOrPut 0L
            var res = 0L
            for (c in 0..m) {
                val perm = (nCr(m, c)*pow(1L*nums[i], 1L*c)) % M
                val sp = f((mask+c)/2, m-c, k-((mask+c) and 1).toInt(), i+1)
                res = (res + (perm * sp) % M)%M
            }
            res
        }
        return f(0, m, k, 0).toInt()
    }

```

