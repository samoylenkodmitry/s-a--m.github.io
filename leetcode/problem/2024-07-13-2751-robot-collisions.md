---
layout: leetcode-entry
title: "2751. Robot Collisions"
permalink: "/leetcode/problem/2024-07-13-2751-robot-collisions/"
leetcode_ui: true
entry_slug: "2024-07-13-2751-robot-collisions"
---

[2751. Robot Collisions](https://leetcode.com/problems/robot-collisions/description/) hard
[blog post](https://leetcode.com/problems/robot-collisions/solutions/5469066/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13072024-2751-robot-collisions?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/wsHxfYzGJPg)
![2024-07-13_09-40_1.webp](/assets/leetcode_daily_images/d68c6515.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/668

#### Problem TLDR

1-D dimensional robots fight #hard #stack

#### Intuition

Sort by positions, then solve the matching parenthesis subproblem. We can use a Stack.

```j

    // 11 44 16
    //  1 20 17
    //  R L  R
    //
    //  1->    17->     <-20
    //  11     16         44

```

#### Approach

* move 'L' as much as possible in a while loop

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun survivedRobotsHealths(positions: IntArray, healths: IntArray, directions: String) =
        with(Stack<Int>()) {
            val inds = positions.indices.sortedBy { positions[it] }
            for (i in inds) if (directions[i] > 'L') push(i) else {
                while (size > 0 && directions[peek()] > 'L')
                    if (healths[peek()] == healths[i]) { pop(); healths[i] = 0; break }
                    else if (healths[peek()] < healths[i]) { pop(); healths[i]-- }
                    else { healths[peek()]--; healths[i] = 0; break }
                if (healths[i] > 0) push(i)
            }
            sorted().map { healths[it] }
        }

```
```rust

    pub fn survived_robots_healths(positions: Vec<i32>, mut healths: Vec<i32>, directions: String) -> Vec<i32> {
        let (mut st, mut inds, d) = (vec![], (0..positions.len()).collect::<Vec<_>>(), directions.as_bytes());
        inds.sort_unstable_by_key(|&i| positions[i]);
        for i in inds {
            if d[i] > b'L' { st.push(i) } else {
                while let Some(&j) = st.last() {
                    if d[j] < b'R' { break }
                    if healths[j] > healths[i] { healths[j] -= 1; healths[i] = 0; break }
                    else if healths[j] < healths[i] { st.pop(); healths[i] -= 1 }
                    else { st.pop(); healths[i] = 0; break }
                }
                if healths[i] > 0 { st.push(i) }
            }
        }
        st.sort_unstable(); st.iter().map(|&i| healths[i]).collect()
    }

```

