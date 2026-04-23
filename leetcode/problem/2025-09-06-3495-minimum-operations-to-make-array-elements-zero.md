---
layout: leetcode-entry
title: "3495. Minimum Operations to Make Array Elements Zero"
permalink: "/leetcode/problem/2025-09-06-3495-minimum-operations-to-make-array-elements-zero/"
leetcode_ui: true
entry_slug: "2025-09-06-3495-minimum-operations-to-make-array-elements-zero"
---

[3495. Minimum Operations to Make Array Elements Zero](https://leetcode.com/problems/minimum-operations-to-make-array-elements-zero/description) hard
[blog post](https://leetcode.com/problems/minimum-operations-to-make-array-elements-zero/solutions/7160918/kotlin-rust-by-samoylenkodmitry-vnav/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/06092025-3495-minimum-operations?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Fg-lZfEfG0Q)

![1.webp](/assets/leetcode_daily_images/7409046c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1104

#### Problem TLDR

Sum query [a..b] of optimal pairwise /4 until range is zero #hard #math

#### Intuition

Didn't solve.

```j
    //  2, 3, 4, 5, 6
    // nums are consequent,
    // there can be a fast way to check ops count

    //              *
    //           *  *
    //        *  *  * ------ divide by four
    //     *  *  *  *
    //  *  *  *  *  *
    //  *  *  *  *  *
    //  0  0  1  1  1

    //    1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
    //    1 1 1 2 2 2 2 2 2 2  2  2  2  2  2  3 ....
    //  1*3+12*2 +
    //  1*(4^1-4^0) + 2*(4^2-4^1) + 3*(4^3-4^2)
    //
    // optimal way?
    // pairs  1 2 3 4 5
    //                  the largest log_4(X) steps
    //    1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
    //      .       .                       3 4
    //      .       .                       0 1
    //      .       .                    3    0
    //      .       .                 3  0
    //      .       .              3  0
    //      .       .           2  0
    //      .       .        2  0
    //      .       .     2  0
    //      .       .   2 0
    //      .       . 1 0
    //      .       1 0
    //      .     1 0
    //      .   1 0
    //      . 0 0
    //    0 0

    //  1 1 1 1 2 2 2 2 2 2  2  2  2  2  2  2  3 3
    //      a                                    b

    //                                     4^3
    //                                     4^2
    //                                     4^1
    //                                     4^0
    //                                     0

    // we have this sequence
    //
    //  1*(4^1-4^0) + 2*(4^2-4^1) + 3*(4^3-4^2) + k*(4^k-4^(k-1))
    //       /                           \
    //      a                             b somewhere there
    //
    //    1 1 1 2 2 2 2 2 2  2  2  2  2  2  2  3 3
    //      a=3     .                           b=17
    //              .                          3..steps
    //              .                    2....steps
    //              .              2....steps
    //              .        2....steps
    //              .   2....steps
    //              2....steps
    //          2....steps
    //      1...step
    // r=1+2*6+3=1+2*(4^2-4^1)/2+3
    // how many steps to take pairs from a to b
    //
    // 1. find which range is b
    // ok let's look for hints (44 minute)
    // first hint already knew - steps(x) = log_4(x)
    // second hint don't understand - pair 2 numbers with max "/4" what?

```

* each number has uniq growing number of ops
* the sequence of ops is `1*(4^1-4^0) + 2*(4^2-4^1) + 3*(4^3-4^2) + k*(4^k-4^(k-1))`
* compute sum of individual ops in range: `ops += (r-l+1)*pow`
* convert to pairwise and handle edge case of odd numbers: `res += ops/2+ops%2`

#### Approach

* i was close to the solution, just didn't believe i'm on a right track;
* the tricky part is converting from singles to pairwise ops; have to do big observation for this, or just give up
* not mine bithack: `0x15555555 >> (30 - 2*p)`evaluates to
`1 + 4 + 4² + … + 4^{p-1}`. Why? `0x15555555` is the bit pattern `0101...` (1s in even positions). Shifting by 30-2p moves exactly p of those 1s into the low end, giving the geometric series above.

Logic behind `(x+1)*p-(4^p-1)/3`:
![image.png](/assets/leetcode_daily_images/99e49177.webp)

For `[a,b]=[20,70]`:
![1.webp](/assets/leetcode_daily_images/c552469c.webp)

#### Complexity

- Time complexity:
$$O(nlog(d))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 8ms
    fun f(x: Int): Long = ((33 - x.countLeadingZeroBits()) shr 1).let { p ->
        1L*(x + 1) * p - (0x15555555 shr (30 - 2 * p)) }
    fun minOperations(qs: Array<IntArray>) = qs.sumOf { (f(it[1])-f(it[0]-1)+1)/2 }

```
```kotlin

// 69ms
    fun minOperations(qs: Array<IntArray>) =
        qs.sumOf { (a,b) ->
            ((1..16).sumOf { p ->
                max(0, 1L*min(b, (1 shl p*2)-1) - max(a, 1 shl (p-1)*2) + 1) * p
            }+1) / 2
        }

```
```rust

// 24ms
    pub fn min_operations(q: Vec<Vec<i32>>) -> i64 {
        q.iter().map(|q| (1+(1..17).map(|p|
            0.max(1 + q[1].min((1<<p*2)-1) - q[0].max(1<<(p-1)*2)) as i64*p
        ).sum::<i64>())/2 ).sum::<i64>()
    }

```
```c++

// 30ms
    long long minOperations(vector<vector<int>>& q) {
        long long r = 0, o = 1;
        for (auto& q: q) {
            for (int p = 1; p < 17; ++p)
            o += p * max(0LL, 1LL + min(1LL*q[1], (1LL<<p*2)-1) - max(1LL*q[0], 1LL<<(p-1)*2));
            r += o / 2, o = 1;
        }
        return r;
    }

```

