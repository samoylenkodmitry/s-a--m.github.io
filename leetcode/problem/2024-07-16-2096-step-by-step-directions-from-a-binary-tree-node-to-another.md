---
layout: leetcode-entry
title: "2096. Step-By-Step Directions From a Binary Tree Node to Another"
permalink: "/leetcode/problem/2024-07-16-2096-step-by-step-directions-from-a-binary-tree-node-to-another/"
leetcode_ui: true
entry_slug: "2024-07-16-2096-step-by-step-directions-from-a-binary-tree-node-to-another"
---

[2096. Step-By-Step Directions From a Binary Tree Node to Another](https://leetcode.com/problems/step-by-step-directions-from-a-binary-tree-node-to-another/description/) medium
[blog post](https://leetcode.com/problems/step-by-step-directions-from-a-binary-tree-node-to-another/solutions/5484259/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/16072024-2096-step-by-step-directions?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/wjZfcaBbt4Y)
![2024-07-16_09-53_1.webp](/assets/leetcode_daily_images/c0981e22.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/672

#### Problem TLDR

`U`-`L`-`R` path from `s` to `d` in a Binary Tree #medium #tree

#### Intuition

My first intuition was to do this in a single Depth-First Search step: if left is found and right is found return the result. However, this gives TLE, as concatenating the path on the fly will worsen the time to O(path^2).

Then I checked the discussion and got the hint to use a StringBuilder. However, this can't be done in a single recursion step, as we should insert to the middle of the path string sometimes.

Now, the working solution: find the Lowest Common Ancestor and go down from it to the `source` and to the `destination`.

Another nice code simplification is achieved by finding two paths from the root, and then removing the common prefix from them.

#### Approach

* we can be careful with StringBuilder removals or just make `toString` on the target

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun getDirections(root: TreeNode?, startValue: Int, destValue: Int): String {
        fun down(n: TreeNode?, v: Int, sb: StringBuilder = StringBuilder()): String = n?.run {
            if (`val` == v) sb.toString() else {
                sb.append('L'); val l = down(left, v, sb)
                if (l != "") l else {
                    sb.deleteAt(sb.lastIndex); sb.append('R')
                    down(right, v, sb).also { sb.deleteAt(sb.lastIndex) }
                }
            }
        } ?: ""
        val s = down(root, startValue); val d = down(root, destValue)
        val skip = s.commonPrefixWith(d).length
        return "U".repeat(s.length - skip) + d.drop(skip)
    }

```
```rust

    pub fn get_directions(root: Option<Rc<RefCell<TreeNode>>>, start_value: i32, dest_value: i32) -> String {
        fn down(n: &Option<Rc<RefCell<TreeNode>>>, v: i32, p: &mut Vec<u8>) -> bool {
            let Some(n) = n else { return false }; let n = n.borrow(); if n.val == v { true } else {
                p.push(b'L'); let l = down(&n.left, v, p);
                if l { true } else {
                    p.pop(); p.push(b'R'); let r = down(&n.right, v, p);
                    if r { true } else { p.pop(); false }
                }
            }
        }
        let mut s = vec![]; let mut d = vec![];
        down(&root, start_value, &mut s); down(&root, dest_value, &mut d);
        let skip = s.iter().zip(d.iter()).take_while(|(s, d)| s == d).count();
        let b: Vec<_> = (0..s.len() - skip).map(|i| b'U').chain(d.into_iter().skip(skip)).collect();
        String::from_utf8(b).unwrap()
    }

```

