using System;
using System.Drawing;
using System.Linq;
using RedUtils;
using RedUtils.Math;

namespace Bot;

public class PartyBot : RUBot
{
    public override void Run()
	{
		Renderer.Text3D((Action != null) ? Action.ToString() : "", Me.Location + Vec3.Up * 30, 1f, Color.Red);

		bool oppBeatsUsToBall = false;
		if (LivingOpponents.Count > 0)
		{
			Car[] ourTeam = new Car[1] { Me }.Concat(LivingTeammates).ToArray();
			Car[] opponents = LivingOpponents.ToArray();
			oppBeatsUsToBall = InterceptCalculator.CalculateInterceptAdvantage(ourTeam, opponents, Ball.MainBall).OpponentsAreFaster;
		}
		if (IsKickoff && Action == null)
		{
			bool doKickoff = true;
			foreach (Car teammate in Teammates)
			{
				if (teammate.Location.Dist(Ball.Location) < Me.Location.Dist(Ball.Location))
				{
					doKickoff = false;
				}
				else if (Me.Location.Dist(Ball.Location) == teammate.Location.Dist(Ball.Location) && (Team != 0 || !(Me.Location.x > teammate.Location.x)) && (Team != 1 || !(Me.Location.x < teammate.Location.x)))
				{
					doKickoff = false;
				}
			}
			if (doKickoff)
			{
				Action = new Kickoff();
			}
			else if (MathF.Abs(Me.Location.x) > 2000f)
			{
				Action = new GetBoost(Me, -1, interruptible: false);
			}
			else
			{
				Action = new Drive(Me, Ball.Location, 1400f);
			}
		}
		else
		{
			if (IsKickoff || (Action != null && ((!(Action is Drive) && !(Action is GetBoost) && !(Action is Arrive)) || !Action.Interruptible)))
			{
				return;
			}
			Shot shot = FindShot(DefaultShotCheck, new Target(TheirGoal));
			if (ShouldBeOffensive())
			{
				if (Me.Boost < 0f)
				{
					Action = new GetBoost(Me);
					return;
				}
				IAction action = shot;
				Action = action ?? Action ?? new Drive(Me, Ball.Location, 2300f, allowDodges: true, wasteBoost: true);
			}
			else if (Me.Boost < 30f)
			{
				Action = new GetBoost(Me);
			}
			else if (!oppBeatsUsToBall)
			{
				Action = shot;
			}
			else
			{
				Vec3 target = Ball.Location + (OurGoal.Location - Ball.Location).Normalize() * 2500f;
				Action = new Drive(Me, target, 1500f, allowDodges: false);
			}
		}
	}

	public bool ShouldBeOffensive()
	{
		if (LivingTeammates.Count == 0)
		{
			return true;
		}
		float biasedDist = Me.Location.FlatDist(Ball.Location) - Me.Boost * 5f;
		foreach (Car livingTeammate in LivingTeammates)
		{
			if (livingTeammate.Location.FlatDist(Ball.Location) - livingTeammate.Boost * 20f < biasedDist)
			{
				return false;
			}
		}
		return true;
	}
}
