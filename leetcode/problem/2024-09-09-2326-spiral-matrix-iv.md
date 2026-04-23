---
layout: leetcode-entry
title: "2326. Spiral Matrix IV"
permalink: "/leetcode/problem/2024-09-09-2326-spiral-matrix-iv/"
leetcode_ui: true
entry_slug: "2024-09-09-2326-spiral-matrix-iv"
---

[2326. Spiral Matrix IV](https://leetcode.com/problems/spiral-matrix-iv/description/) medium
[blog post](https://leetcode.com/problems/spiral-matrix-iv/solutions/5758866/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/09092024-2326-spiral-matrix-iv?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/2sQQiX8fmd0)
![1.webp](/assets/leetcode_daily_images/5fda5c2d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/729

#### Problem TLDR

LinkedList to spiral 2D matrix #medium #linked_list #simulation

#### Intuition

The only tricky thing is the implementation. Use the values themselves to detect when to change the direction.

#### Approach

* only one single rotation per cycle is necessary
* use 2D vector rotation: `(dx dy) = (-dy dx)`

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(nm)$$

#### Code

```kotlin

    fun spiralMatrix(m: Int, n: Int, head: ListNode?): Array<IntArray> {
        val res = Array(m) { IntArray(n) { -1 }}
        var y = 0; var x = 0; var curr = head; var dy = 0; var dx = 1
        while (curr != null) {
            res[y][x] = curr.`val`
            curr = curr.next
            if ((x + dx) !in 0..<n || (y + dy) !in 0..<m || res[y + dy][x + dx] >= 0)
                dx = -dy.also { dy = dx }
            x += dx; y += dy
        }
        return res
    }

```
```rust

    pub fn spiral_matrix(m: i32, n: i32, mut head: Option<Box<ListNode>>) -> Vec<Vec<i32>> {
        let mut res = vec![vec![-1; n as usize]; m as usize];
        let (mut y, mut x, mut dy, mut dx) = (0, 0, 0i32, 1i32);
        while let Some(head_box) = head {
            res[y as usize][x as usize] = head_box.val; head = head_box.next;
            if x < -dx || y < -dy || x + dx >= n || y + dy >= m || res[(y + dy) as usize][(x + dx) as usize] >= 0 {
                (dx, dy) = (-dy, dx)
            }
            x += dx; y += dy
        }
        res
    }

```
```c++

    vector<vector<int>> spiralMatrix(int m, int n, ListNode* head) {
        vector<vector<int>> res(m, vector(n, -1)); int y = 0; int x = 0;
        int dy = 0; int dx = 1;
        for (; head; head = head->next) {
            res[y][x] = head->val;
            if (x < -dx || y < -dy || x + dx >= n || y + dy >= m || res[y + dy][x + dx] >= 0) {
                std::swap(dx, dy); dx *= -1;
            }
            x += dx; y += dy;
        }
        return res;
    }

```

