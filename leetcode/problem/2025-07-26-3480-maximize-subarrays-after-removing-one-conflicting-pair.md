---
layout: leetcode-entry
title: "3480. Maximize Subarrays After Removing One Conflicting Pair"
permalink: "/leetcode/problem/2025-07-26-3480-maximize-subarrays-after-removing-one-conflicting-pair/"
leetcode_ui: true
entry_slug: "2025-07-26-3480-maximize-subarrays-after-removing-one-conflicting-pair"
---

[3480. Maximize Subarrays After Removing One Conflicting Pair](https://leetcode.com/problems/maximize-subarrays-after-removing-one-conflicting-pair/description/) hard
[blog post](https://leetcode.com/problems/maximize-subarrays-after-removing-one-conflicting-pair/solutions/7006658/kotlin-by-samoylenkodmitry-5e4i/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26072025-3480-maximize-subarrays?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/2l1T5xRRJxQ)
![1.webp](/assets/leetcode_daily_images/8737f504.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1061

#### Problem TLDR

Max subarrays count without excluded pairs-1 #hard #greedy #line_sweep

#### Intuition

Didn't solved.
```j
    // 1 2 3 4
    // a     b
    //   a b

    // 1 2 3 4 5
    // a b
    //   a     b
    //     a   b

    // subproblem: nuber of subarrays with a and b separated
    // 1 2 3 4 5 6 7 8 9
    //     a     b
    //
    // for a: can't go past b, so it is number subarrays in 1..5 1..b-1
    // 1 2 3 4 5, 12 23 34 45, 123 234 345, 1234 2345, 12345
    // 5 + 4 + 3 + 2 + 1 = 5*(5 + 1)/2=15
    //
    // for b: can't go before a, 4..9  a+1..n, 6*7/2=21
    //
    // overlap: 4 5, 45 = b-a-1 = 2 * 3 /2 = 3
    //
    // subarrays 15+21-3=33  (b-1)*b/2 + (n-a)*(n-a+1)/2 - (b-a-1)*(b-a)/2
    //                        n^2/2-na+n/2-a+ab
    //
    // reverse the problem: number of subarrays including a and b
    // f(a) + f(n-b)
    // all is f(n)

    // now what if we have two pairs?
    // 1 2 3 4 5 6 7 8 9
    //       a       b
    //   a       b
    // . . . . .            valid range
    //         . . . . .    valid range
    //       a   b          is the pair intersection result (28 minute)
    //
    // what if pairs are not intersecting
    // 1 2 3 4 5 6 7 8 9
    //   a   b
    //           a   b
    // . . .              valid
    //             . . .  valid
    //     . . . . .      valid
    //
    // more complex
    // . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    //   a     b
    //     a       b
    //                   a       b
    //                       a       b
    //                                     a     b
    //                                                 a     b
    // . . . .
    //     . . . .
    //       . . . . . . . . . .
    // ..and so on, i use the hint at (36 minute)
    // then i gave up and look for solution https://leetcode.com/problems/maximize-subarrays-after-removing-one-conflicting-pair/solutions/6527930/intuitive-solution-with-sorting-visually-explained/
    // so, initial idea to sort was right
    // then, how to scan and compute?
    // we look for intervals between prev_max_a..max_a and tail after current b
    //  12345678901234567890123
    //  a      b              n
    //      a      b          n
    //          a       b     n
    //  .      ................
    //  .....      ............ c += (max1-max2)*tail = (5-1)*(23-8+1)
    //      .....       .......
    // overlapping case:
    //      a  b
    //  a          b
    //  .....  ................
    //   ....      ............
    // total excluded is sum(max(a) * b_tail)
    // current excluded is sum_overlaping(a1-a2) * b_tail
    // we adding back maximum excluded value (while keeping track of overlaps)
```

#### Approach

* the overlapping part is the hardest to understand

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 450ms
    fun maxSubarrays(n: Int, p: Array<IntArray>): Long {
        for (p in p) if (p[0] > p[1]) p[0] = p[1].also { p[1] = p[0] }
        Arrays.sort(p, compareBy{ it[1] })
        var max1 = 0; var max2 = 0; var c = 0L; var maxc = 0L; var exc = 0L
        for (i in p.indices) {
            val (a, b) = p[i]; var tail = (if (i < p.size - 1) p[i + 1][1] else n + 1) - b
            if (a > max1) { max2 = max1; max1 = a; c = 0 } else max2 = max(max2, a)
            c += 1L * (max1 - max2) * tail
            exc += 1L * max1 * tail
            maxc = max(maxc, c)
        }
        return 1L * n * (n + 1) / 2 - exc + maxc
    }

```

