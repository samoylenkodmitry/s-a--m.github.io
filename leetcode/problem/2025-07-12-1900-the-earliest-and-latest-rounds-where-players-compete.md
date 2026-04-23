---
layout: leetcode-entry
title: "1900. The Earliest and Latest Rounds Where Players Compete"
permalink: "/leetcode/problem/2025-07-12-1900-the-earliest-and-latest-rounds-where-players-compete/"
leetcode_ui: true
entry_slug: "2025-07-12-1900-the-earliest-and-latest-rounds-where-players-compete"
---

[1900. The Earliest and Latest Rounds Where Players Compete](https://leetcode.com/problems/the-earliest-and-latest-rounds-where-players-compete/description) hard
[blog post](https://leetcode.com/problems/the-earliest-and-latest-rounds-where-players-compete/solutions/6949047/kotlin-by-samoylenkodmitry-mihd/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12072025-1900-the-earliest-and-latest?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/S9JrZQjncdU)
![1.webp](/assets/leetcode_daily_images/9951652d.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1047

#### Problem TLDR

First and last round Alice fight Bob #hard #simulation

#### Intuition

Didn't solved (have a hard time to understand the simulation rules)

```j
    // 7 minutes read description, didn't understood
    // let's try simulation
    // round 1
    // 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    // *              s               w   choose 11win, 6middle go 2nd rnd
    // 2, 3, 4, 5, 7, 8, 9, 10
    // w                     *            2 always win
    // 2, 3, 4, 5, 7, 8, 9
    // w        s        *                2 always win, 5 middle go 2nd rnd
    // 2, 3, 4, 7, 8
    // w     s     *                      2 win, 4 go 2nd
    // 2 3 7
    // w s *                              2 win, 3 go 2nd
    // 2                                  2 go 2nd

    // round 2
    // 6 11 2 5 4 3
    // 2 3 4 5 6 11
    // let's start with simulation 28^28
    // how to choose the best?
    // maybe BFS (idea on 51 minute)
    // 1:38 wrong answer for case
    // 1 2 3 4   (2,3)            3,3  vs 1,1
    // w     -
    //   2 3
    //     w
    //   2 3        wrong simulation code
    // is winnder goes fight again? (question at 1:43 :)  )

    // 1, 2, 3, 4, 5, 6  7, 8, 9, 10, 11
    // -              s               w
    //    2, 3, 4, 5,    7, 8, 9, 10
    //    w                       -
    //       3, 4, 5,    7, 8, 9
    //       w                 -
    //          4, 5,    7, 8
    //          w           -
    //             5,    7
    //             w
    //             5,    7

    // 1 2 3 4    (2,3)
    // w     -
    ```

What went wrong: the description comprehension. We stop when first fight with second. The winners are irrelevant. My mistake was leaving the winner.

#### Approach

* sometimes the description is the hardest part

#### Complexity

- Time complexity:
$$O(2^n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 343ms
    fun earliestAndLatest(n: Int, fp: Int, sp: Int): IntArray {
        var min = Int.MAX_VALUE; var max = 1; val fp = fp - 1; val sp = sp - 1
        fun dfs(mask: Int, round: Int, i: Int, j: Int) {
            if (i >= j) dfs(mask, round + 1, 0, 27) else
            if ((mask and (1 shl i)) == 0) dfs(mask, round, i + 1, j) else
            if ((mask and (1 shl j)) == 0) dfs(mask, round, i, j - 1) else
            if (i == fp && j == sp) { min = min(min, round); max = max(max, round) } else {
                if (i != fp && i != sp) dfs(mask xor (1 shl i), round, i + 1, j - 1)
                if (j != fp && j != sp) dfs(mask xor (1 shl j), round, i + 1, j - 1)
            }
        }
        dfs((1 shl n) - 1, 1, 0, 27)
        return intArrayOf(min, max)
    }

```

