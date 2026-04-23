---
layout: leetcode-entry
title: "2929. Distribute Candies Among Children II"
permalink: "/leetcode/problem/2025-06-01-2929-distribute-candies-among-children-ii/"
leetcode_ui: true
entry_slug: "2025-06-01-2929-distribute-candies-among-children-ii"
---

[2929. Distribute Candies Among Children II](https://leetcode.com/problems/distribute-candies-among-children-ii/description/) medium
[blog post](https://leetcode.com/problems/distribute-candies-among-children-ii/solutions/6801343/kotlin-rust-by-samoylenkodmitry-c29k/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/01062025-2929-distribute-candies?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/XuIjEQX5uXo)
![1.webp](/assets/leetcode_daily_images/6735afdb.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1006

#### Problem TLDR

Ways to give `limited` from `n` candies to 3 kids #medium #math #combinations

#### Intuition

Didn't solved this, as hint4 gave the exact answer.

Useful hint instead of hint4:
* for each first kid loked `i = 0..min(n, limit)`
* do a running sum
* of allowed range that both second and third kid can take
* n - i - limit .. n - i
* trim the range: max(0, n - i - limit)..min(limit, n - i)
* the length of range a..b = b - a + 1

Chain-of-thoughts:

```j

    // 10^6 (should  be faster then linear?)
    // is this completely math problem? like Cr(a, b)?
    // 4 hints, acceptance rate <50% (shold be hard problem?)
    // ok let's think how the distribution works:
    // 5 candles, limit 2
    // always 3 chilren
    // | A | B | C |
    // (0..limit) | (0..limit) | (0..limit)
    // the number of ways is countA * countB * countC
    // is n <= 3 * limit ? (let's run test case n=4, limit = 1, yes, the number of ways is 0)
    // no, do we have to handle the symmetry:
    // 0..limit | 0..min(n - countA, limit) | 0..min(n - countA - countB, limit)
    // consider n = 5 limit = 2
    // 1 2 2
    // 2 1 2
    // 2 2 1 symmetry with 1 2 2, counts separately
    //
    // so, we have 3 numbers,
    // A=0..min(n, limit)
    // B=0..min(n - A, limit)
    // C=0..min(n - A - B, limit)
    // the result is A * B * C (is this correct?, no, b=0, c=0)
    // should we do a 3-step dfs dp? (10^6 will give TLE, but let's try)
    // ok, dp works, but TLE
    // probably requires some math idea from combinatorics
    // let's look for hints (22 minute)
    // enumerate first 0..min(n, limit) (already knew)
    // second is 0..j..limit, i + j less n (interesting way to write this)
    //                        j less n - i
    //           0..min(n - i, limit) (already knew)
    // hint: "after some transformations"
    // basically give you the answer on the hint4

```

There is a math solution:
1. total `n` stars and `2` bars (to separate candies to three kids: * * | * * | * * )
2. trick is how to count `invalid` combinations
3. there is a math theory for this https://en.wikipedia.org/wiki/Inclusion%E2%80%93exclusion_principle
4. good explanation: https://leetcode.com/problems/distribute-candies-among-children-ii/solutions/4278816/o-1-combinatorics/

Let's try to summarize explanation for counding the invalid combinations:
1. one kid has more than a limit, (limit + 1), 3 kids, remove `limit + 1` and count ways to set bars C(n - (limit+1), 2), call those combinations as `A`
2. two kids has more than a limit, (limit + 1), 3 pairs of kids, remove `2 * (limit + 1)` and count ways to set bars C(n-2*(limit+1), 2), call those combinations as `B`
3. three kids has more than a limit, single tripple of kids, remove `3 * (limit + 1)`, count ways to set bars C(n-3(limit+1), 2)
4. `important trick`: the `A` combinations are including the `B` combinations already, so we have to subract the `B` (weak point)
5. some math for stars and bars:

```j
//  1 2 3 4 5   | |
// . . . . . .
// (n+1) positions, choose 2
// C(n+1, 2) = n!/(r!*(n-r)!) = (n+1)!/(2!*(n+1-2)!) = (n+1)*n*(n-1)!/(2 *(n-1)!) = n(n+1)/2
```

#### Approach

* try to understand the combinatorics, it seems the level has been raised to require this
* c++ solution for this

#### Complexity
- Time complexity:
$$O(n)$$ or O(1) if you are genius

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 123ms
    fun distributeCandies(n: Int, limit: Int) =
        (0..min(n, limit)).sumOf { i ->
            1L * max(0, min(limit, n - i) - max(0, n - i - limit) + 1) }

```
```rust

// 15ms
    pub fn distribute_candies(n: i32, limit: i32) -> i64 {
        (0..=n.min(limit)).map(|i|
            0.max(limit.min(n - i) - 0.max(n - i - limit) + 1) as i64
        ).sum()
    }

```
```c++

// 0ms
    long long distributeCandies(int n, int limit) {
        auto c = [&](this const auto& c, long n) -> long {return n*(n+1)/2;};
        n++; // stars are places between candies to put the bars
        long long nStarsTwoBars = 1LL * c(n);
        long long oneOutOfLimit = 1LL * max(0, n - (limit + 1));
        long long twoOutOfLimit = max(0, n - 2 * (limit + 1));
        long long threeOutOfLimit = max(0, n - 3 * (limit + 1));
        long long invalidCombinations = 3 * c(oneOutOfLimit)
                                       -3 * c(twoOutOfLimit)
                                       +1 * c(threeOutOfLimit);
        return nStarsTwoBars - invalidCombinations;
    }

```

