---
layout: leetcode-entry
title: "42. Trapping Rain Water"
permalink: "/leetcode/problem/2024-04-12-42-trapping-rain-water/"
leetcode_ui: true
entry_slug: "2024-04-12-42-trapping-rain-water"
---

[42. Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/description/) hard
[blog post](https://leetcode.com/problems/trapping-rain-water/solutions/5010867/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12042024-42-trapping-rain-water?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/uYfidIUHm94)
![2024-04-12_08-45.webp](/assets/leetcode_daily_images/b74c5ccb.webp)

#### Problem TLDR

Trap the water in area between vertical walls #hard

#### Intuition

Let's observe some examples and try to apply `decreasing stack` technique somehow:

```j
    //               #
    //       #       # #   #
    //   #   # #   # # # # # #
    //i0 1 2 3 4 5 6 7 8 91011
    // 0 1 0 2 1 0 1 3 2 1 2 1
    //0*             .          0(0)
    //1  *           .          1
    //2    *         .          1(1) 0(2)
    //3      *       .          2(3)     + (3-2)*(1-0)
    //4        *     .          2(3) 1(4)
    //5          *   .          2(3) 1(4) 0(5)
    //6            * .          2(3) 1(6)    + (1-0)*(5-4)
    //7              *          3(7)         + (2-1)*(6-3)

    //2#  #
    //1## #
    //0####
    // 0123
    //
    // 0 1 2 3
    // 2 1 0 2
    // *          2(0)
    //   *        2(0) 1(1)
    //     *      2(0) 1(1) 0(2)
    //       *    2(3)           + a=2,b=1, (i-b-1)*(h[b]-h[a])=(3-1-1)*(1-0)
    //                             a=1,b=0, (3-0-1)*(2-1)

    // #
    // #   #
    // # # #
    // # # #
    // 0 1 2
    // 4 2 3

    //           #
    // #         #
    // #     #   #
    // # #   # # #
    // # #   # # #
    // 0 1 2 3 4 5
    // 4 2 0 3 2 5

    // #         #
    // #         #
    // #         #
    // # #   #   #
    // # # # # # #
    // 0 1 2 3 4 5
    // 5 2 1 2 1 5
```
As we meet a new high value we can collect some water. There are corner cases when the left border is smaller than the right.

#### Approach

* try to come up with as many corner cases as possible
* horizontal width must be between the highest columns

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun trap(height: IntArray): Int = with(Stack<Int>()) {
        var sum = 0
        for ((i, hb) in height.withIndex()) {
            while (size > 0 && height[peek()] <= hb) {
                val ha = height[pop()]
                if (size > 0) sum += (i - peek() - 1) * (min(hb, height[peek()]) - ha)
            }
            push(i)
        }
        return sum
    }

```
```rust

    pub fn trap(height: Vec<i32>) -> i32 {
        let (mut sum, mut stack) = (0, vec![]);
        for (i, &hb) in height.iter().enumerate() {
            while stack.len() > 0 && height[*stack.last().unwrap()] <= hb {
                let ha = height[stack.pop().unwrap()];
                if stack.len() > 0 {
                    let dh = hb.min(height[*stack.last().unwrap()]) - ha;
                    sum += ((i - *stack.last().unwrap()) as i32 - 1) * dh
                }
            }
            stack.push(i)
        }
        sum
    }

```

