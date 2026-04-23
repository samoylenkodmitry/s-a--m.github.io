---
layout: leetcode-entry
title: "3666. Minimum Operations to Equalize Binary String"
permalink: "/leetcode/problem/2026-02-27-3666-minimum-operations-to-equalize-binary-string/"
leetcode_ui: true
entry_slug: "2026-02-27-3666-minimum-operations-to-equalize-binary-string"
---

[3666. Minimum Operations to Equalize Binary String](https://open.substack.com/pub/dmitriisamoilenko/p/27022026-3666-minimum-operations?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) hard
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/27022026-3666-minimum-operations?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27022026-3666-minimum-operations?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/PBxyeNXKqag)

![img](/assets/leetcode_daily_images/7e72480f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1282

#### Problem TLDR

Min steps to convert 01 stirng to ones flipping K #hard #bfs

#### Intuition

Didn't solve.
```j
 // acceptance rate 40%
    // i will give up after 30 minutes
    //
    //
    // 110    k=1
    //   1
    //
    // 0101   k=3
    // 10 0
    //  111
    //
    // 101    k=2
    // 01
    //  00
    // 11
    // ...    -1
    //
    // there should be a law or shortcut?
    //
    // simulation - impossible
    //
    // all k should be selected
    // order doesnt matter
    // so it is just a two numbers: zeros and ones
    //
    // 00000001111111    k
    //
    // how many zeros to choose at each step?
    // dp? try all counts?
    //
    // how to detect that we can't reach the end?
    //
    // z 5 o 5 k=6
    // o += min(k,z)   o=5+5=10
    // o -= k-min(k,z) o=10-1=9
    // z = k-min(k,z)  z=1
    //
    //
    // 9minute
    //
    // z 1 o 9 k=6  so, this can loop forever
    //              any strategy that would work?
    // 5:5 - 1:9 - 7:3 - BFS? numbers up to 10^5
    //                   each round is n choices
    //                   prune with visited set
    //
    //                   i have no other ideas, let's try
    // 14 ;minute
    // 20 minute - wrong answer 0101 k=3 my is 1, expected 2
    // 26 minute - wrong answer 001 k=3 my is 2, expected -1
    // 32 minute - wrong answer 000 k = 1, my is -1, expected 3
    //             so my entire intuition doesnt work on the repeated cases
    //             how to mitigate?
    //             k=1, zeros = 3, repeats = z/k = 3
    //             so this should be dijkstra with steps?
    // anyway i can be wrong and already spent 35 minutes, lets' go hints
```

The TLE BFS: number of zeros are the state, next states are max(0,k-ones)..min(k,zeros).
The optimization: parity hack, from z=5 k=3 we can jump to exactly any of 2,4,6,8. Same parity, continuous (steps 2), strict L..R range. Now, instead of individual state jumps consider range adjustments L..R.
We want the L to be close to 0, R to be close to N. So range grows continously, peek only interesting range.

For L to reach 0, we want it be close to K, and also want L..R be wider to skip-jump to K. For R reach N, we target R-K to flip all bits.

#### Approach

* use ai

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
// 13ms
    fun minOperations(s: String, k: Int): Int {
        var L = s.count { it == '0' }; var R = L
        for (step in 0..s.length) {
            if (L == 0) return step
            val newL = when {
                k in L..R -> (L + k) % 2
                k < L -> L - k
                else -> k - R
            }
            val targetR = s.length - k
            val newR = s.length - when {
                targetR in L..R -> (L + targetR) % 2
                targetR < L -> L - targetR
                else -> targetR - R
            }
            L = newL; R = newR
        }
        return -1
    }
```

